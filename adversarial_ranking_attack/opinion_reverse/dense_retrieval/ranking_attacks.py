import argparse
import bisect
import os
import sys
from collections import defaultdict

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import torch
import tqdm
# from pattern.text.en import singularize, pluralize
from transformers import BertTokenizer
import numpy as np

#from ir.bert_models import BertForConcatNextSentencePrediction, BertForLM
from ir.scorer import SentenceScorer
from ir.collision_util.constraints_utils import create_constraints, get_sub_masks, get_inputs_filter_ids, STOPWORDS
from ir.collision_util.logging_utils import log
from ir.collision_util.optimization_utils import perturb_logits
from ir.collision_util.tokenizer_utils import valid_tokenization
from opinion_reverse.dense_retrieval.condenser import condenser_loss, condenser_encode
from nltk.corpus import stopwords

K = 10
BIRCH_DIR = prodir + '/data/birch'
BIRCH_MODEL_DIR = BIRCH_DIR + '/models'
BIRCH_DATA_DIR = BIRCH_DIR + '/data'
BIRCH_ALPHAS = [1.0, 0.5, 0.1]
BIRCH_GAMMA = 0.6
BERT_LM_MODEL_DIR = prodir + '/data/wiki103/bert'
BOS_TOKEN = '[unused0]'
device_cpu = torch.device("cpu")

"""
collision/PAT for dense retrieval models
"""

def find_filters(query, model, tokenizer, device, k=500):
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words], pad_to_max_length=True)
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
    n = len(words)
    batch_size = 512
    n_batches = n // batch_size + 1
    all_scores = []
    for i in tqdm.trange(n_batches, desc='Filtering vocab'):
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
        outputs = model.forward(input_ids, attention_masks, token_type_ids)
        scores = outputs[0][:, 1]
        all_scores.append(scores)

    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]


def add_single_plural(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    contains = []
    for word in tokenizer.vocab:
        if word.isalpha() and len(word) > 2:
            for t in tokens:
                if len(t) > 2 and word != t and (word.startswith(t) or t.startswith(word)):
                    contains.append(word)

    for t in tokens[:]:
        if not t.isalpha():
            continue
        sig_t = singularize(t)
        plu_t = pluralize(t)
        if sig_t != t and sig_t in tokenizer.vocab:
            tokens.append(sig_t)
        if plu_t != t and plu_t in tokenizer.vocab:
            tokens.append(plu_t)

    return [w for w in tokens + contains if w not in STOPWORDS]


def gen_aggressive_collision(inputs_a, inputs_b, model, tokenizer, device, margin=None, lm_model=None, args=None):
    word_embedding = model.get_input_embeddings().weight.detach()
    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    input_mask = torch.zeros(vocab_size, device=device)
    # filters = find_filters(inputs_a, model, tokenizer, device, k=args.num_filters)
    # 419
    # best_ids = get_inputs_filter_ids(inputs_b, tokenizer)
    # input_mask[best_ids] = 0.68
    # remove_tokens = add_single_plural(inputs_a, tokenizer)
    # if args.verbose:
    #     print(','.join(remove_tokens))

    # remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    # remove_ids.append(tokenizer.vocab['.'])
    # input_mask[remove_ids] = 0.68

    # num_filters_ids = tokenizer.convert_tokens_to_ids(filters)
    # input_mask[num_filters_ids] = 0.68

    sub_mask = get_sub_masks(tokenizer, device)

    query_ids = tokenizer.encode(inputs_a)
    query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)
    # prevent output num_filters neighbor words
    seq_len = args.seq_len
    # [50, q_len + 2]
    batch_query_ids = torch.cat([query_ids] * args.topk, 0)
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        masked_x = x + input_mask + sub_mask
        if args.regularize:
            masked_x += stopwords_mask
        p = torch.softmax(masked_x / args.stemp, -1)
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        x = torch.cat([x, word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]
    
    def ids_to_emb(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb

    # some constants
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    if args.target == "nb_bert":
        labels = np.array([[0,1] for t in range(1)])
        labels = torch.tensor(labels, dtype=torch.float, device=device)
    elif args.target == 'bge':
        labels = np.array([1.0 for t in range(1)])
        labels = torch.tensor(labels, dtype=torch.float, device=device)
    else:
        labels = torch.ones((1,), dtype=torch.float16, device=device)
    repetition_penalty = 1.0

    best_collision = None
    best_score = -1e9
    prev_score = -1e9
    collision_cands = []
    patience = 0

    var_size = (seq_len, vocab_size)
    z_i = torch.zeros(*var_size, requires_grad=True, device=device)
    # z_i = torch.normal(mean=0., std=1.0, size=var_size, requires_grad=True, device=device)
    for it in range(args.max_iter):
        optimizer = torch.optim.Adam([z_i], lr=args.lr)
        for j in range(args.perturb_iter):
            optimizer.zero_grad()
            # relaxation
            p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
            # forward to BERT with relaxed inputs
            query_emb = ids_to_emb(query_ids)

            concat_inputs_emb = torch.cat([query_emb, inputs_embeds], dim=1)

            if args.target == "mini":
                outputs = model(
                    inputs_embeds = concat_inputs_emb,
                    labels=labels
                )
                loss, cls_logits = outputs[0], outputs[1]
                cls_logits_score = cls_logits[:, 0]
            elif args.target == "nb_bert":
                outputs = model(
                    inputs_embeds = concat_inputs_emb,
                    labels=labels
                )
                loss, cls_logits = outputs[0], outputs[1]
                cls_logits_score = cls_logits[:, 1]
            elif args.target == "mini_adv" or args.target == "nb_bert_adv":

                outputs = model(
                    inputs_embeds_pos=concat_inputs_emb,
                    inputs_embeds_neg=concat_inputs_emb,
                    labels=torch.tensor(labels, dtype=torch.int64)
                )
                loss, cls_logits = outputs[0], outputs[1]
                cls_logits_score = cls_logits[:, 1]
            elif args.target == 'bge':
                outputs = model(
                    query_emb=query_emb,
                    passage_emb=inputs_embeds,
                    labels=labels,
                )
                loss ,cls_logits = outputs[0], outputs[1]
                cls_logits_score = cls_logits[:]
            
            if margin is not None:
                loss += torch.sum(torch.relu(margin - cls_logits_score))

            if args.beta > 0.:
                lm_loss = get_lm_loss(p_inputs)
                loss = args.beta * lm_loss + (1 - args.beta) * loss
            loss.backward()
            optimizer.step()
            if args.verbose and (j + 1) % 10 == 0:
                print(f'It{it}-{j + 1}, loss={loss.item()}')

        # detach to free GPU memory
        z_i = z_i.detach()
        _, topk_tokens = torch.topk(z_i, args.topk)
        probs_i = torch.softmax(z_i / args.stemp, -1).unsqueeze(0).expand(args.topk, seq_len, vocab_size)

        output_so_far = None
        # beam search left to right, get cand collisions
        for t in range(seq_len):
            t_topk_tokens = topk_tokens[t]
            t_topk_onehot = torch.nn.functional.one_hot(t_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(args.num_beams):
                next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
                if output_so_far is None:
                    context = probs_i.clone()
                else:
                    output_len = output_so_far.shape[1]
                    beam_topk_output = output_so_far[j].unsqueeze(0).expand(args.topk, output_len)
                    beam_topk_output = torch.nn.functional.one_hot(beam_topk_output, vocab_size)
                    context = torch.cat([beam_topk_output.float(), probs_i[:, output_len:].clone()], 1)
                context[:, t] = t_topk_onehot
                context_embeds = torch.einsum('blv,vh->blh', context, word_embedding)
                context_embeds = torch.cat([context_embeds, batch_sep_embeds], 1)
                batch_query_emb = ids_to_emb(batch_query_ids)
                concat_batch_inputs_emb = torch.cat([batch_query_emb, context_embeds], dim=1)
                if args.target == "mini" or args.target == "nb_bert":
                    outputs_1 = model(
                        inputs_embeds = concat_batch_inputs_emb,
                    )
                    clf_logits = outputs_1[0]
                    clf_logits_score = clf_logits[:, 0]
                elif args.target == "nb_bert":
                    outputs = model(
                        inputs_embeds = concat_batch_inputs_emb,
                    )
                    clf_logits = outputs_1[0]
                    clf_logits_score = clf_logits[:, 1]
                elif args.target == "mini_adv" or args.target == "nb_bert_adv":
                    outputs_1 = model(
                        inputs_embeds_pos=concat_batch_inputs_emb,
                        inputs_embeds_neg=concat_batch_inputs_emb,
                    )
                    clf_logits = outputs_1[0]
                    clf_logits_score = clf_logits[:, 1]
                elif args.target == 'bge':
                    outputs_1 = model(
                        query_emb=batch_query_emb,
                        passage_emb=context_embeds,
                    )
                    clf_logits = outputs_1[0]
                    clf_logits_score = clf_logits[:]
                #clf_logits = model(inputs_embeds=concat_batch_inputs_emb)[0]
                #clf_scores = clf_logits[:, 0].detach().float()
                clf_scores = clf_logits_score.detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask

            if args.regularize:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9

            if output_so_far is not None and repetition_penalty > 1.0:
                lm_model.enforce_repetition_penalty_(next_scores, 1, args.num_beams, output_so_far, repetition_penalty)

            # re-organize to group the beam together
            # (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(1, args.num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, args.num_beams, dim=1, largest=True, sorted=True)
            # next batch beam content
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[0], next_scores[0])):
                # get beam and token IDs
                # beam_id = beam_token_id // vocab_size
                beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='trunc')
                token_id = beam_token_id % vocab_size
                next_sent_beam.append((beam_token_score, token_id, beam_id))

            next_batch_beam = next_sent_beam
            # sanity check / prepare next batch
            assert len(next_batch_beam) == args.num_beams
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)
            # re-order batch
            if output_so_far is None:
                output_so_far = beam_tokens.unsqueeze(1)
            else:
                output_so_far = output_so_far[beam_idx, :]
                output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)

        pad_output_so_far = torch.cat([output_so_far, sep_tensor[:args.num_beams].unsqueeze(1)], 1)
        concat_query_ids = torch.cat([batch_query_ids[:args.num_beams], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:args.num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1)

        if args.target == "mini":
            outputs_2 = model(
                input_ids=concat_query_ids, 
                token_type_ids=token_type_ids
            )
            clf_logits = outputs_2[0]
            clf_logits_score = clf_logits[:, 0]
        elif args.target == "nb_bert":
            outputs_2 = model(
                input_ids=concat_query_ids, 
                token_type_ids=token_type_ids
            )
            clf_logits = outputs_2[0]
            clf_logits_score = clf_logits[:, 1]
        elif args.target == "mini_adv" or args.target == "nb_bert_adv":
            outputs_2 = model(
                input_ids_pos=concat_query_ids,
                input_ids_neg=concat_query_ids,
                token_type_ids_pos=token_type_ids,
                token_type_ids_neg=token_type_ids,
            )
            clf_logits = outputs_2[0]
            clf_logits_score = clf_logits[:, 1]
        elif args.target == "bge":
            outputs_2 = model(
                query_id=batch_query_ids[:args.num_beams],
                passage_id=pad_output_so_far,
            )
            clf_logits = outputs_2[0]
            clf_logits_score = clf_logits[:]
        #clf_logits = model(input_ids=concat_query_ids, token_type_ids=token_type_ids)[0]
        actual_clf_scores = clf_logits_score#
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]

        # for valiadation the output
        valid_idx = sorter[0]
        valid = True
        # valid = False
        # for idx in sorter:
        #     valid, _ = valid_tokenization(output_so_far[idx], tokenizer)
        #     if valid:
        #         valid_idx = idx
        #         break

        # re-initialize z_i
        curr_best = output_so_far[valid_idx]
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = 0.1
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)

        curr_score = actual_clf_scores[valid_idx].item()
        if valid and curr_score > best_score:
            patience = 0
            best_score = curr_score
            best_collision = tokenizer.decode(curr_best.cpu().tolist())
            print(curr_score)

        if curr_score <= prev_score:
            # break
            patience += 1
        if patience > args.patience_limit:
            break
        prev_score = curr_score

    return best_collision, best_score, collision_cands

def gen_natural_collision_DR(inputs_a, inputs_b, model, tokenizer, device, lm_model, margin=None, eval_lm_model=None, args=None):
    input_mask = torch.zeros(tokenizer.vocab_size, device=device)
    #filters = find_filters(inputs_a, model, tokenizer, device, k=args.num_filters)
    # best_ids = get_inputs_filter_ids(inputs_b, tokenizer)
    # input_mask[best_ids] = 0.1

    # num_filters_ids = tokenizer.convert_tokens_to_ids(filters)
    # input_mask[num_filters_ids] = 0.1

    # remove_tokens = add_single_plural(inputs_a, tokenizer)
    # if args.verbose:
    #     log(','.join(remove_tokens))
    # remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    # input_mask[remove_ids] = -1e9
    input_mask[tokenizer.convert_tokens_to_ids(['.', '@', '='])] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9
    #filter_ids = [tokenizer.vocab[w] for w in tokenizer.vocab if not w.isalnum()]
    vocab = tokenizer.vocab
    filter_ids = [vocab[w] for w in vocab if not w.isalnum()]
    first_mask = torch.zeros_like(input_mask)
    first_mask[filter_ids] = -1e9
    
    collition_init = tokenizer.convert_tokens_to_ids([BOS_TOKEN])
    start_idx = 1
    num_beams = args.num_beams
    repetition_penalty = 5.0
    curr_len = len(collition_init)

    # scores for each sentence in the beam
    beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
    beam_scores[1:] = -1e9

    output_so_far = torch.tensor([collition_init] * num_beams, device=device)
    past = None
    vocab_size = tokenizer.vocab_size
    topk = args.topk
    query_ids = tokenizer.encode(inputs_a)

    query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)
    batch_query_ids = torch.cat([query_ids] * topk, 0)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)

    is_first = True
    word_embedding = model.get_input_embeddings().weight.detach()
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    # batch_sep_embeds = word_embedding[sep_tensor]
    if args.target == "nb_bert":
        batch_labels = np.array([[0,1] for t in range(num_beams)])
        batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    elif args.target == "bge" or args.target == "condenser":
        batch_labels = np.array([1 for t in range(num_beams)])
        batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    else:
        batch_labels = torch.ones((num_beams,), dtype=torch.float, device=device)


    def ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()

        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding.cpu())
        return input_emb.to(device)
    
    def ids_to_emb_with_model(input_ids, dense_model):
        if args.target == 'bge':
            input_emb = dense_model.bge_inference({'input_ids':input_ids, 'attention_mask':None})['dense_vecs']
        elif args.target == 'condenser':
            input_emb = dense_model.encode(inputs_a)
            input_emb = torch.cat([input_emb] * num_beams, 0)
        else:
            input_emb = dense_model(input_ids)[0][:,0,:]
        return input_emb
    
    # batch_query_emb = ids_to_emb(batch_query_ids[:num_beams])
    batch_query_emb = ids_to_emb_with_model(batch_query_ids[:num_beams], model)

    def classifier_loss(p, context, target):#context:
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], 1)
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding)
        # add embeddings for SEP
        x = torch.cat([x, batch_sep_embeds[:num_beams]], 1)
        concat_input_emb = torch.cat([batch_query_emb.unsqueeze(1), x], dim=1)
        if target == "mini" or target =="nb_bert":
            output = model(
                    inputs_embeds = concat_input_emb,
                    labels=batch_labels
                )
            cls_loss = output[0]
        elif target == "condenser":
            # x_emb = model(inputs_embeds = x)[0][:,0,:]
            x_emb = model.encode_with_emb(x)
            cls_loss = condenser_loss(
                query_emb = torch.nn.functional.normalize(batch_query_emb, p=2, dim=1),
                passage_emb = torch.nn.functional.normalize(x_emb, p=2, dim=1),
                labels = batch_labels,
                device = device,
                args = args
            )
        elif target =="bge":
            output = model(
                query_emb = batch_query_emb.unsqueeze(1),
                passage_emb = x,
                labels = batch_labels
            )
            cls_loss = output[0]
        #cls_loss = model(inputs_embeds=concat_input_emb, labels=batch_labels)[0]
        return cls_loss

    best_score = -1e9
    best_collision = None
    collision_cands = []

    while (curr_len - start_idx) < args.seq_len:
        model_inputs = lm_model.prepare_inputs_for_generation(output_so_far, past=past)
        outputs = lm_model(**model_inputs)
        present = outputs[1]
        # (batch_size * num_beams, vocab_size)
        next_token_logits = outputs[0][:, -1, :]
        # next_token_logits.to(torch.device("cpu"))
        
        lm_scores = torch.log_softmax(next_token_logits, dim=-1)
        if args.perturb_iter > 0:
            # perturb internal states of LM
            def target_model_wrapper(p):
                return classifier_loss(p, output_so_far.detach()[:, start_idx:], args.target)
            
            next_token_logits = perturb_logits(
                next_token_logits,
                args.lr,
                target_model_wrapper,
                num_iterations=args.perturb_iter,
                kl_scale=args.kl_scale,
                temperature=args.stemp,
                device=device,
                verbose=args.verbose,
                logit_mask=input_mask,
            )
        
        if repetition_penalty > 1.0:
            lm_model.enforce_repetition_penalty_(next_token_logits, 1, num_beams,  output_so_far, repetition_penalty)
        next_token_logits = next_token_logits / args.stemp

        # (batch_size * num_beams, vocab_size)
        next_lm_scores = lm_scores + beam_scores[:, None].expand_as(lm_scores)
        _, topk_tokens = torch.topk(next_token_logits, topk)
        # get target model score here
        next_clf_scores = []
        for i in range(num_beams):
            next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if output_so_far.shape[1] > start_idx:
                curr_beam_topk = output_so_far[i, start_idx:].unsqueeze(0).expand(
                    topk, output_so_far.shape[1] - start_idx)
                # (topk, curr_len + next_token + sep)
                curr_beam_topk = torch.cat([curr_beam_topk, topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)
            else:
                curr_beam_topk = torch.cat([topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)
            concat_query_ids = torch.cat([batch_query_ids, curr_beam_topk], 1)
            token_type_ids = torch.cat([torch.zeros_like(batch_query_ids), torch.ones_like(curr_beam_topk), ], 1)
            if args.target == "mini":
                outputs_1 = model(
                    input_ids=concat_query_ids,
                    token_type_ids=token_type_ids
                )  
                clf_logits = outputs_1[0]
                clf_scores = torch.log_softmax(clf_logits, -1)[:, 0].detach()
            elif args.target == "nb_bert":
                outputs_1 = model(
                    input_ids=concat_query_ids,
                    token_type_ids=token_type_ids
                )  
                clf_logits = outputs_1[0]
                clf_scores = torch.log_softmax(clf_logits, -1)[:, 1].detach()
            elif args.target == "condenser":
                outputs_1 = condenser_encode(
                    model,
                    concat_query_ids,
                    curr_beam_topk,
                    device,
                    args
                )
                clf_logits = outputs_1[0]
                clf_scores = torch.log_softmax(clf_logits, -1).detach()
            elif args.target == "bge":
                outputs_1 = model(
                    query_id=batch_query_ids,
                    passage_id = curr_beam_topk,
                )  
                clf_logits = outputs_1[0]
                clf_scores = torch.log_softmax(clf_logits, -1).detach()
            #clf_logits = model(input_ids=concat_query_ids, token_type_ids=token_type_ids)[0]
            #clf_scores = torch.log_softmax(clf_logits, -1)[:, 0].detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))
        
        next_clf_scores = torch.cat(next_clf_scores, 0)

        if is_first:
            next_clf_scores += beam_scores[:, None].expand_as(lm_scores)
            next_clf_scores += first_mask
            is_first = False


        next_scores = (1 - args.beta) * next_clf_scores + args.beta * next_lm_scores
        next_scores += input_mask
        
        # re-organize to group the beam together
        # (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(num_beams * vocab_size)
        next_lm_scores = next_lm_scores.view(num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, num_beams, largest=True, sorted=True)

        next_lm_scores = next_lm_scores[next_tokens]
        # next batch beam content
        next_sent_beam = []

        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens, next_lm_scores)):
            # get beam and token IDs
            beam_id = beam_token_id // vocab_size
            token_id = beam_token_id % vocab_size
            next_sent_beam.append((beam_token_score, token_id, beam_id))

        next_batch_beam = next_sent_beam

        # sanity check / prepare next batch
        assert len(next_batch_beam) == num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = output_so_far.new([x[1] for x in next_batch_beam])
        beam_idx = output_so_far.new([x[2] for x in next_batch_beam])

        # re-order batch
        output_so_far = output_so_far[beam_idx, :]
        output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)
        
        # sanity check
        pad_output_so_far = torch.cat([output_so_far[:, start_idx:], sep_tensor[:num_beams].unsqueeze(1)], 1)
        concat_query_ids = torch.cat([batch_query_ids[:num_beams], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1)
        if args.target == "mini":
            outputs_2 = model(
                input_ids=concat_query_ids,
                token_type_ids=token_type_ids
            )  
            clf_logits = outputs_2[0]
            actual_clf_scores = clf_logits[:, 0]
        elif args.target == "nb_bert":
            outputs_2 = model(
                input_ids=concat_query_ids,
                token_type_ids=token_type_ids
            )  
            clf_logits = outputs_2[0]
            actual_clf_scores = clf_logits[:, 1]
        elif args.target == "condenser":
            outputs_2 = condenser_encode(
                model,
                concat_query_ids,
                pad_output_so_far,
                device,
                args
            )
            clf_logits = outputs_2[0]
            actual_clf_scores = clf_logits
        elif args.target == 'bge':
            outputs_2 = model(
                query_id=batch_query_ids[:num_beams],
                passage_id = pad_output_so_far,
            )
            clf_logits = outputs_2[0]
            actual_clf_scores = clf_logits
        #clf_logits = model(input_ids=concat_query_ids, token_type_ids=token_type_ids)[0]
        #actual_clf_scores = clf_logits[:, 0]
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i, start_idx:].cpu().tolist())}'
                for i in sorter
            ]
            log(f'Margin={margin if margin else 0:.4f}, query={inputs_a} | ' + ' | '.join(decoded))

        if curr_len > args.min_len:
            valid_idx = sorter[0]
            valid = True
            # valid = False
            # for idx in sorter:
            #     valid, _ = valid_tokenization(output_so_far[idx, start_idx:], tokenizer)
            #     if valid:
            #         valid_idx = idx
            #         break
            
            curr_score = actual_clf_scores[valid_idx].item()
            curr_collision = tokenizer.decode(output_so_far[valid_idx, start_idx:].cpu().tolist())
            collision_cands.append((curr_score, curr_collision))
            if valid and curr_score > best_score:
                best_score = curr_score
                best_collision = curr_collision
            print("CUR:", curr_score, curr_collision,"BEST:", best_score)

            if args.verbose:
                lm_perp = eval_lm_model.perplexity(curr_collision)
                log(f'LM perp={lm_perp.item()}')
        
        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # update current length
        curr_len = curr_len + 1

    return best_collision, best_score, collision_cands

def get_inputs_sim_ids(inputs, tokenizer):
    tokens = [w for w in tokenizer.tokenize(inputs) if w.isalpha() and w not in set(stopwords.words('english'))]
    return tokenizer.convert_tokens_to_ids(tokens)


def find_sims(query, model, tokenizer, device, args, words, k=300):
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words], padding=True)
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    n = len(words)
    batch_size = 1024
    n_batches = n // batch_size + 1
    all_scores = []
    
    for i in range(n_batches):
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        outputs = model(input_ids_pos=input_ids,
                            token_type_ids_pos=token_type_ids,
                            input_ids_neg=input_ids,
                            token_type_ids_neg=token_type_ids)
        scores = outputs[0][:, 1]
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    sims = set([words[i.item()] for i in top_indices])
    return [w for w in sims if w.isalpha()]

def find_sims_DR(query, model, tokenizer, device, args, words, k=300):
    inputs_query = tokenizer.batch_encode_plus([query for w in words], padding=True)
    inputs_words = tokenizer.batch_encode_plus([w for w in words], padding=True)
    all_input_ids_query = torch.tensor(inputs_query['input_ids'], device=device)
    all_input_ids_words = torch.tensor(inputs_words['input_ids'], device=device)
    all_token_type_ids_query = torch.tensor(inputs_query['token_type_ids'], device=device)
    all_token_type_ids_words = torch.tensor(inputs_words['token_type_ids'], device=device)
    n = len(words)
    batch_size = 1024
    n_batches = n // batch_size + 1
    all_scores = []
    
    for i in range(n_batches):
        input_ids_query = all_input_ids_query[i * batch_size: (i + 1) * batch_size]
        input_ids_words = all_input_ids_words[i * batch_size: (i + 1) * batch_size]
        token_type_ids_query = all_token_type_ids_query[i * batch_size: (i + 1) * batch_size]
        token_type_ids_words = all_token_type_ids_words[i * batch_size: (i + 1) * batch_size]
        query = {'input_ids':input_ids_query, 'token_type_ids':token_type_ids_query}
        words_ = {'input_ids':input_ids_words, 'token_type_ids':token_type_ids_words}
        outputs = model(query=query,
                            pos=words_,
                            neg=words_,
                            )
        scores = outputs[0]
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    sims = set([words[i.item()] for i in top_indices])
    return [w for w in sims if w.isalpha()]

def logits_perturbation(
        unpert_logits,
        lr=0.001,
        target_model_wrapper=None,
        max_iter=5,
        temperature=1.0,
        device="cuda",
        logit_mask=None,
):
    # initialize perturbation variable
    perturbation = torch.tensor(np.zeros(unpert_logits.shape, dtype=np.float32), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        logits = unpert_logits * temperature + perturbation + logit_mask
        probs = torch.softmax(logits / temperature, -1)

        loss = torch.scalar_tensor(0.0).to(device)
        loss_list = []

        if target_model_wrapper is not None:
            discrim_loss = target_model_wrapper(probs)
            loss += discrim_loss
            loss_list.append(discrim_loss)

        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    # apply perturbations
    pert_logits = unpert_logits * temperature + perturbation
    return pert_logits

def dict_device(input, device):
    if 'input_ids'in input:
        input['input_ids'] = input['input_ids'].to(device)
    if 'attention_mask' in input:
        input['attention_mask'] = input['attention_mask'].to(device)
    if 'token_type_ids' in input:
        input['token_type_ids'] = input['token_type_ids'].to(device)
    return input

def pairwise_anchor_trigger_DR(query, anchor, raw_passage, model, tokenizer, device, words, lm_model=None, args=None,
                            nsp_model=None):
    print("QUERY:", query)
    input_mask = torch.zeros(tokenizer.vocab_size, device=device)
    sims = find_sims_DR(query, model, tokenizer, device, args, words=words, k=args.num_sims)

    best_ids = get_inputs_sim_ids(anchor, tokenizer)
    input_mask[best_ids] = 0.68
    
    num_sims_ids = tokenizer.convert_tokens_to_ids(sims)
    input_mask[num_sims_ids] = 0.68

    input_mask[tokenizer.convert_tokens_to_ids(['.', '@', '='])] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9
    
    vocab = tokenizer.vocab
    sim_ids = [vocab[w] for w in vocab if not w.isalnum()]
    first_mask = torch.zeros_like(input_mask)
    first_mask[sim_ids] = -1e9

    trigger_init = tokenizer.convert_tokens_to_ids([BOS_TOKEN])
    start_idx = 1
    num_beams = args.num_beams
    repetition_penalty = 5.0
    curr_len = len(trigger_init)

    beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
    beam_scores[1:] = -1e9

    output_so_far = torch.tensor([trigger_init] * num_beams, device=device)
    
    past = None
    vocab_size = tokenizer.vocab_size
    topk = args.topk
    #QUERY PROCESS
    query_token = tokenizer(query, max_length=args.max_seq_len,padding="max_length", truncation=True, add_special_tokens=True,return_tensors='pt')
    query_ids = query_token['input_ids']#, add_special_tokens=True
    # query_ids = query_ids.unsqueeze(0)
    query_attention = query_token['attention_mask']
    query_token_type = query_token['token_type_ids']
    query_ids_ = {'input_ids':query_ids, 'attention_mask':query_attention}

    batch_query_ids = {'input_ids':torch.cat([query_ids] * topk, 0), 'attention_mask':torch.cat([query_attention] * topk, 0), 'token_type_ids':torch.cat([query_token_type] * topk, 0)}
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)
    period_tensor = torch.tensor([1012] * topk, device=device)
    cls_tensor = torch.tensor([101] * topk, device=device)

    is_first = True
    word_embedding = model.get_input_embeddings().weight.detach().cpu()
    # prevent waste GPU memory in one-hot transformation
    word_embedding_cuda = model.get_input_embeddings().weight.detach()
    passage_ids_no_pad = tokenizer(raw_passage, max_length=args.max_seq_len)
    passage_ids_no_pad_cat = torch.tensor([passage_ids_no_pad['input_ids']] * topk, device=device)
    passage_ids = tokenizer(raw_passage,  max_length=args.max_seq_len,padding="max_length", truncation=True, add_special_tokens=True)['input_ids']
    passage_mask = tokenizer(raw_passage,  max_length=args.max_seq_len,padding="max_length", truncation=True, add_special_tokens=True)['attention_mask']
    passage_ids_tensor = torch.tensor([passage_ids] * num_beams, device=device)
    passage_oh = torch.nn.functional.one_hot(passage_ids_no_pad_cat, len(word_embedding))
    if args.pat:
        nsp_word_embedding = nsp_model.get_input_embeddings().weight.detach().cpu()
        nsp_word_embedding_cuda = nsp_model.get_input_embeddings().weight.detach()
    #
    batch_sep_embeds = word_embedding_cuda[sep_tensor].unsqueeze(1)
    batch_labels = torch.ones((num_beams,), dtype=torch.long, device=device_cpu)
    # batch_labels = np.array([[0,1] for t in range(num_beams)])
    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device_cpu)

    anchor_token = tokenizer(anchor, max_length=args.max_seq_len,padding="max_length", truncation=True ,add_special_tokens=True, return_tensors='pt')
    anchor_ids = anchor_token['input_ids']
    anchor_attention = anchor_token['attention_mask']
    
    batch_anchor_ids = {'input_ids': torch.cat([anchor_ids[:128]] * args.topk, 0), 'attention_mask':torch.cat([anchor_attention[:128]] * args.topk, 0)}

    def ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb.to(device)

    def ids_to_emb_with_model(input_ids, dense_model):
        # input_ids = {'input_ids':input_ids}
        # input_emb = dense_model.condenser(**input_ids)[0][:,0,:].squeeze()
        input_emb = dense_model.encode_(input_ids)
        return input_emb
    
    def nsp_ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, nsp_word_embedding)
        return input_emb.to(device)


    if args.nsp:
        # raw passage transform to ids,只有一个passage
        passage_ids = tokenizer.encode(raw_passage, add_special_tokens=True)
        # do not remove [CLS]
        passage_ids = torch.tensor(passage_ids[:args.max_seq_len]).unsqueeze(0)
        batch_passage_ids = torch.cat([passage_ids] * args.topk, 0)
        batch_passage_embds = nsp_ids_to_emb(batch_passage_ids)

    batch_query_emb = ids_to_emb_with_model(dict_device(batch_query_ids, device), model)
    batch_anchor_emb = ids_to_emb_with_model(dict_device(batch_anchor_ids, device), model)
    # concat_inputs_emb_neg = torch.cat([batch_query_emb, batch_anchor_emb], dim=1)

    def classifier_loss(p, context, passage_oh):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], dim=1)#
        one_hot = torch.cat([one_hot, passage_oh[:num_beams]], dim=1)
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding_cuda)
        # add [SEP]
        x = torch.cat([x, batch_sep_embeds[:num_beams]], dim=1)
        # concat_inputs_emb_pos = torch.cat([batch_query_emb[:num_beams], x], dim=1)
        x_emb = model.condenser(inputs_embeds = x)[0][:,0,:].squeeze()
        outputs = model(
            inputs_embeds_query = batch_query_emb[:num_beams, :],
            inputs_embeds_pos=x_emb,
            # inputs_embeds_neg=batch_anchor_emb[:num_beams, :128, :],
            inputs_embeds_neg=batch_anchor_emb[:num_beams, :],
            labels=batch_labels
        )
        cls_loss = outputs[0]
        return cls_loss

    best_score = -1e9
    best_trigger = None
    trigger_cands = []
    while (curr_len - start_idx) < args.seq_len:
        model_inputs = lm_model.prepare_inputs_for_generation(output_so_far, past=past)
        outputs = lm_model(**model_inputs)
        present = outputs[1]
        # [B * Beams, V]
        next_token_logits = outputs[0][:, -1, :]
        lm_scores = torch.log_softmax(next_token_logits, dim=-1)
        if args.perturb_iter > 0:
            # perturb internal states of LM
            def target_model_wrapper(p):
                return classifier_loss(p, output_so_far.detach()[:, start_idx:], passage_oh)

            next_token_logits = logits_perturbation(
                next_token_logits,
                lr=args.lr,
                target_model_wrapper=target_model_wrapper,
                max_iter=args.perturb_iter,
                temperature=args.stemp,
                device=device,
                logit_mask=input_mask,
            )
        
        if repetition_penalty > 1.0:
            lm_model.enforce_repetition_penalty_(next_token_logits, 1, num_beams, output_so_far, repetition_penalty)
        next_token_logits = next_token_logits / args.stemp
        
        # [B * Beams, V]
        next_lm_scores = lm_scores + beam_scores[:, None].expand_as(lm_scores)
        _, topk_tokens = torch.topk(next_token_logits, topk)

        next_clf_scores = []
        next_nsp_scores = []
        for i in range(num_beams):
            next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if args.nsp:
                next_beam_nsp_losses = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
            if output_so_far.shape[1] > start_idx:
                curr_beam_topk = output_so_far[i, start_idx:].unsqueeze(0).expand(topk,
                                                                                  output_so_far.shape[1] - start_idx)
                # [topk, curr_len + next_token + sep]
                curr_beam_topk = torch.cat([curr_beam_topk, topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)
            else:
                curr_beam_topk = torch.cat([topk_tokens[i].unsqueeze(1), sep_tensor.unsqueeze(1)], 1)

            concat_input_ids_pos = torch.cat([curr_beam_topk, passage_ids_no_pad_cat], 1)
            # token_type_ids_pos = torch.cat([torch.zeros_like(batch_query_ids), torch.ones_like(curr_beam_topk)], 1)
            clf_logits = model(
                query=batch_query_ids,
                # token_type_ids_query=token_type_ids_pos,
                pos={'input_ids':concat_input_ids_pos},#, 'attention_mask': torch.ones_like(curr_beam_topk)
                # token_type_ids_pos=token_type_ids_pos,
                neg={'input_ids':concat_input_ids_pos},
                # token_type_ids_neg=token_type_ids_pos
            )[0]
            clf_scores = torch.log_softmax(clf_logits, -1).detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))

            if args.nsp:
                concat_nsp_embs = torch.cat([
                    nsp_ids_to_emb(curr_beam_topk), batch_passage_embds,], dim=1)[:, :128, :]
                nsp_logits = nsp_model(inputs_embeds=concat_nsp_embs, return_dict=True)["logits"]
                # 0 indicates sequence B is a continuation of sequence A,
                nsp_scores = torch.log_softmax(nsp_logits, -1)[:, 1].detach()
                next_beam_nsp_losses.scatter_(0, topk_tokens[i], nsp_scores.float())
                next_nsp_scores.append(next_beam_nsp_losses.unsqueeze(0))
        next_clf_scores = torch.cat(next_clf_scores, 0)
        if args.nsp:
            next_nsp_scores = torch.cat(next_nsp_scores)
        
        if is_first:
            next_clf_scores += beam_scores[:, None].expand_as(lm_scores)
            next_clf_scores += first_mask
            if args.nsp:
                next_nsp_scores += beam_scores[:, None].expand_as(lm_scores)
                next_nsp_scores += first_mask
            is_first = False

        # attack loss
        if args.nsp:
            next_scores = next_clf_scores + args.lambda_1 * next_lm_scores - args.lambda_2 * next_nsp_scores
        else:
            next_scores = next_clf_scores + args.lambda_1 * next_lm_scores
        next_scores += input_mask

        # re-organize to group the beam together
        # keepping top triggers accross beams
        next_scores = next_scores.view(num_beams * vocab_size)
        next_lm_scores = next_lm_scores.view(num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, num_beams, largest=True, sorted=True)
        next_lm_scores = next_lm_scores[next_tokens]

        # next batch beam content
        next_sent_beam = []
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens, next_lm_scores)):
            # get beam and token IDs
            beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='trunc')
            token_id = beam_token_id % vocab_size
            next_sent_beam.append((beam_token_score, token_id, beam_id))

        next_batch_beam = next_sent_beam

        assert len(next_batch_beam) == num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = output_so_far.new([x[1] for x in next_batch_beam])
        beam_idx = output_so_far.new([x[2] for x in next_batch_beam])

        # re-order batch
        output_so_far = output_so_far[beam_idx, :]
        output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)

        # sanity check
        pad_output_so_far = torch.cat([cls_tensor[:num_beams].unsqueeze(1), output_so_far[:, start_idx:]], 1)
        trigger_length = pad_output_so_far.shape[1] - 1
        pad_output_so_far = torch.cat([pad_output_so_far, passage_ids_tensor[:num_beams, start_idx:]], 1)
        pad_output_so_far = torch.cat([pad_output_so_far, sep_tensor[:num_beams].unsqueeze(1)], 1)
        if 0 not in passage_mask:
            attention_mask_passage = torch.tensor([passage_mask+[1]*(trigger_length+1)] * num_beams)
            token_type_ids_passage = torch.zeros_like(attention_mask_passage)
        else: 
            passage_id_length = passage_mask.index(0)
            attention_mask_passage = torch.cat([torch.ones([num_beams , trigger_length+passage_id_length]),torch.zeros([num_beams, pad_output_so_far.shape[1] - trigger_length-passage_id_length])], 1)
            token_type_ids_passage = torch.zeros_like(pad_output_so_far)
        
        # concat_query_ids = torch.cat([batch_query_ids[:num_beams], pad_output_so_far], 1)
        # token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:num_beams]),torch.ones_like(pad_output_so_far)], 1)
        final_clf_logits = model(
            query={'input_ids':batch_query_ids['input_ids'][:num_beams], 'attention_mask':batch_query_ids['attention_mask'][:num_beams], 'token_type_ids':batch_query_ids['token_type_ids'][:num_beams]},
            pos={'input_ids':pad_output_so_far, 'attention_mask':attention_mask_passage, 'token_type_ids': token_type_ids_passage},
            # token_type_ids_pos=token_type_ids,
            neg={'input_ids':pad_output_so_far, 'attention_mask':attention_mask_passage, 'token_type_ids': token_type_ids_passage},
            # token_type_ids_neg=token_type_ids
        )[0]
        final_clf_scores = final_clf_logits
        sorter = torch.argsort(final_clf_scores, -1, descending=True)

        curr_score = final_clf_scores[sorter[0]].item()
        curr_trigger = tokenizer.decode(output_so_far[sorter[0], start_idx:].cpu().tolist())
        trigger_cands.append((curr_score, curr_trigger))
        if curr_score > best_score:
            best_score = curr_score
            best_trigger = curr_trigger
        print("BEST_TRI:", best_trigger, best_score)

        # query_ids_t = torch.tensor(query_ids, device=device).unsqueeze(0)
        twithp = best_trigger + " " + raw_passage
        # query_enco = tokenizer(query, truncation=True, return_tensors='pt')['input_ids'].to(device)
        # passage_enco = tokenizer.encode(twithp)
        passage_enco = tokenizer(twithp, max_length=args.max_seq_len, padding = 'max_length',truncation=True, add_special_tokens=True,return_tensors='pt')
        # passage_enco = torch.tensor(passage_enco, device=device)
        cz_out = model(
            query=query_ids_,
            pos = passage_enco,
            neg = passage_enco,
        )
        print("CZ:", cz_out[0])
        
        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # next
        curr_len = curr_len + 1

    return best_trigger, best_score, trigger_cands


