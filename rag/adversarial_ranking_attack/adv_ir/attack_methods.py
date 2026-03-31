import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import numpy as np
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords

BOS_TOKEN = '[unused0]'


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

        loss.backward()
        optimizer.step()

    # apply perturbations
    pert_logits = unpert_logits * temperature + perturbation
    return pert_logits

def pairwise_anchor_trigger(query, anchor, raw_passage, model, tokenizer, device, words, lm_model=None, args=None,
                            nsp_model=None):
    input_mask = torch.zeros(tokenizer.vocab_size, device=device)
    sims = find_sims(query, model, tokenizer, device, args, words=words, k=args.num_sims)
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
    query_ids = tokenizer.encode(query, add_special_tokens=True)

    query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)
    batch_query_ids = torch.cat([query_ids] * topk, 0)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)

    is_first = True
    word_embedding = model.get_input_embeddings().weight.detach().cpu()
    # prevent waste GPU memory in one-hot transformation
    word_embedding_cuda = model.get_input_embeddings().weight.detach()
    if args.pat:
        nsp_word_embedding = nsp_model.get_input_embeddings().weight.detach().cpu()
        nsp_word_embedding_cuda = nsp_model.get_input_embeddings().weight.detach()
    #
    batch_sep_embeds = word_embedding_cuda[sep_tensor].unsqueeze(1)
    batch_labels = torch.ones((num_beams,), dtype=torch.long, device=device)

    anchor_ids = torch.tensor(tokenizer.encode(anchor, add_special_tokens=True)[1:]).unsqueeze(0)
    batch_anchor_ids = torch.cat([anchor_ids[:128]] * args.topk, 0)

    def ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb.to(device)
    
    def nsp_ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, nsp_word_embedding)
        return input_emb.to(device)

    if args.nsp:
        # raw passage transform to ids
        passage_ids = tokenizer.encode(raw_passage, add_special_tokens=True)
        # do not remove [CLS]
        passage_ids = torch.tensor(passage_ids[:128]).unsqueeze(0)
        batch_passage_ids = torch.cat([passage_ids] * args.topk, 0)
        batch_passage_embds = nsp_ids_to_emb(batch_passage_ids)

    batch_query_emb = ids_to_emb(batch_query_ids)
    batch_anchor_emb = ids_to_emb(batch_anchor_ids)
    concat_inputs_emb_neg = torch.cat([batch_query_emb, batch_anchor_emb], dim=1)

    def classifier_loss(p, context):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], dim=1)#
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding_cuda)
        # add [SEP]
        x = torch.cat([x, batch_sep_embeds[:num_beams]], dim=1)
        concat_inputs_emb_pos = torch.cat([batch_query_emb[:num_beams], x], dim=1)
        
        outputs = model(
            inputs_embeds_pos=concat_inputs_emb_pos,
            inputs_embeds_neg=concat_inputs_emb_neg[:num_beams, :128, :],
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
                return classifier_loss(p, output_so_far.detach()[:, start_idx:])

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

            concat_input_ids_pos = torch.cat([batch_query_ids, curr_beam_topk], 1)
            token_type_ids_pos = torch.cat([torch.zeros_like(batch_query_ids), torch.ones_like(curr_beam_topk)], 1)

            clf_logits = model(
                input_ids_pos=concat_input_ids_pos,
                token_type_ids_pos=token_type_ids_pos,
                input_ids_neg=concat_input_ids_pos,
                token_type_ids_neg=token_type_ids_pos
            )[0]
            clf_scores = torch.log_softmax(clf_logits, -1)[:, 1].detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))

            if args.nsp:
                concat_nsp_embs = torch.cat([
                    batch_passage_embds,
                    nsp_ids_to_emb(curr_beam_topk)], dim=1)[:, :128, :]
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
        pad_output_so_far = torch.cat([output_so_far[:, start_idx:], sep_tensor[:num_beams].unsqueeze(1)], 1)
        concat_query_ids = torch.cat([batch_query_ids[:num_beams], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1)
        # [num_beams, 2]
        final_clf_logits = model(
            input_ids_pos=concat_query_ids,
            token_type_ids_pos=token_type_ids,
            input_ids_neg=concat_query_ids,
            token_type_ids_neg=token_type_ids
        )[0]
        final_clf_scores = final_clf_logits[:, 1]
        sorter = torch.argsort(final_clf_scores, -1, descending=True)

        curr_score = final_clf_scores[sorter[0]].item()
        curr_trigger = tokenizer.decode(output_so_far[sorter[0], start_idx:].cpu().tolist())
        trigger_cands.append((curr_score, curr_trigger))
        if curr_score > best_score:
            best_score = curr_score
            best_trigger = curr_trigger
        print("BEST_TRI:", best_trigger, "SCORE:", best_score)

        #RANK test
        twithp = best_trigger + " " + raw_passage
        concat_pos = tokenizer([(query, twithp)],  max_length=args.max_seq_len, padding="max_length", truncation=True, return_tensors='pt').to(device)
        cz_out = model(
            input_ids_pos=concat_pos['input_ids'],
            token_type_ids_pos=concat_pos['token_type_ids'],
            attention_mask_pos=concat_pos['attention_mask'],
            input_ids_neg=concat_pos['input_ids'],
            token_type_ids_neg=concat_pos['token_type_ids'],
            attention_mask_neg=concat_pos['attention_mask'],
        )
        print("CZ:", cz_out[0])

        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # next
        curr_len = curr_len + 1

    return best_trigger, best_score, trigger_cands

def pairwise_anchor_trigger_for_im(query, anchor, raw_passage, model, tokenizer, device, words, lm_model=None, args=None,
                            nsp_model=None):
    print("all_LEN:", len(query)+len(raw_passage))
    input_mask = torch.zeros(tokenizer.vocab_size, device=device)
    sims = find_sims(query, model, tokenizer, device, args, words=words, k=args.num_sims)
    # print("SIM:",sims)
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
    query_ids = tokenizer.encode(query, add_special_tokens=True)

    query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)
    batch_query_ids = torch.cat([query_ids] * topk, 0)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * topk, device=device)
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
    passage_masks_tensor = torch.tensor([passage_mask] * num_beams, device=device)
    if args.pat:
        nsp_word_embedding = nsp_model.get_input_embeddings().weight.detach().cpu()
        nsp_word_embedding_cuda = nsp_model.get_input_embeddings().weight.detach()
    #
    batch_sep_embeds = word_embedding_cuda[sep_tensor].unsqueeze(1)
    batch_labels = torch.ones((num_beams,), dtype=torch.long, device=device)

    anchor_ids = torch.tensor(tokenizer.encode(anchor, add_special_tokens=True)[1:]).unsqueeze(0)
    batch_anchor_ids = torch.cat([anchor_ids[:args.max_seq_len]] * args.topk, 0)

    def ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb.to(device)
    
    def nsp_ids_to_emb(input_ids):
        input_ids = input_ids.clone().detach().cpu()
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, nsp_word_embedding)
        return input_emb.to(device)

    if args.nsp:
        # raw passage transform to ids
        passage_ids = tokenizer.encode(raw_passage, add_special_tokens=True)
        # do not remove [CLS]
        passage_ids = torch.tensor(passage_ids[:args.max_seq_len]).unsqueeze(0)
        batch_passage_ids = torch.cat([passage_ids] * args.topk, 0)
        batch_passage_embds = nsp_ids_to_emb(batch_passage_ids)

    batch_query_emb = ids_to_emb(batch_query_ids)
    batch_anchor_emb = ids_to_emb(batch_anchor_ids)
    concat_inputs_emb_neg = torch.cat([batch_query_emb, batch_anchor_emb], dim=1)

    def classifier_loss(p, context):
        context = torch.nn.functional.one_hot(context, len(word_embedding))
        one_hot = torch.cat([context.float(), p.unsqueeze(1)], dim=1)#
        x = torch.einsum('blv,vh->blh', one_hot, word_embedding_cuda)
        # add [SEP]
        x = torch.cat([x, batch_sep_embeds[:num_beams]], dim=1)
        concat_inputs_emb_pos = torch.cat([batch_query_emb[:num_beams], x], dim=1)
        
        outputs = model(
            inputs_embeds_pos=concat_inputs_emb_pos,
            inputs_embeds_neg=concat_inputs_emb_neg[:num_beams, :128, :],
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
                return classifier_loss(p, output_so_far.detach()[:, start_idx:])

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

            concat_input_ids_pos = torch.cat([batch_query_ids, curr_beam_topk], 1)
            # concat_input_ids_pos = torch.cat([concat_input_ids_pos, passage_ids_no_pad_cat], 1)
            token_type_ids_pos = torch.cat([torch.zeros_like(batch_query_ids), torch.ones_like(curr_beam_topk)], 1)
            # token_type_ids_pos =torch.cat([token_type_ids_pos, torch.ones_like(passage_ids_no_pad_cat)],1)

            clf_logits = model(
                input_ids_pos=concat_input_ids_pos,
                token_type_ids_pos=token_type_ids_pos,
                input_ids_neg=concat_input_ids_pos,
                token_type_ids_neg=token_type_ids_pos
            )[0]
            clf_scores = torch.log_softmax(clf_logits, -1)[:, 1].detach()
            next_beam_scores.scatter_(0, topk_tokens[i], clf_scores.float())
            next_clf_scores.append(next_beam_scores.unsqueeze(0))

            if args.nsp:
                concat_nsp_embs = torch.cat([
                    batch_passage_embds,
                    nsp_ids_to_emb(curr_beam_topk)], dim=1)[:, :128, :]
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
        # pad_output_so_far = torch.cat([output_so_far[:, start_idx:], sep_tensor[:num_beams].unsqueeze(1)], 1)
        # trigger_length = pad_output_so_far.shape[1] - 1
        # pad_output_so_far = torch.cat([cls_tensor[:num_beams].unsqueeze(1), output_so_far[:, start_idx:]], 1)
        # trigger_length = pad_output_so_far.shape[1] - 1
        trigger_id_length = output_so_far[:, start_idx:].shape[1]
        pad_output_so_far = torch.cat([output_so_far[:, start_idx:], passage_ids_tensor[:num_beams, start_idx:]], 1)
        pad_output_so_far = torch.cat([pad_output_so_far, sep_tensor[:num_beams].unsqueeze(1)], 1)
        concat_query_ids = torch.cat([batch_query_ids[:num_beams], pad_output_so_far], 1)
        query_id_length = batch_query_ids.shape[1]
        passage_id_length = passage_mask.index(0) if 0 in passage_mask else len(passage_mask)
        if trigger_id_length+query_id_length+passage_id_length>=args.max_seq_len:
            token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:num_beams]),
                                    torch.ones_like(pad_output_so_far)], 1).to(device)
            attention_mask_pos = torch.ones([num_beams, concat_query_ids.shape[1]]).to(device)
        else:
            token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:num_beams].cpu()),
                                    torch.ones([num_beams, trigger_id_length+passage_id_length])], 1)
            token_type_ids = torch.cat([token_type_ids, torch.zeros([num_beams, concat_query_ids.shape[1]-(trigger_id_length+query_id_length+passage_id_length)])],1).int().to(device)
            attention_mask_pos = torch.cat([torch.ones([num_beams, trigger_id_length+query_id_length+passage_id_length]), torch.zeros([num_beams, concat_query_ids.shape[1]-(trigger_id_length+query_id_length+passage_id_length)])], 1).int().to(device)
        
        concat_query_ids = concat_query_ids[:,:args.max_seq_len]
        token_type_ids = token_type_ids[:,:args.max_seq_len]
        attention_mask_pos = attention_mask_pos[:,:args.max_seq_len]

        # [num_beams, 2]；
        final_clf_logits = model(
            input_ids_pos=concat_query_ids,
            token_type_ids_pos=token_type_ids,
            attention_mask_pos = attention_mask_pos,
            input_ids_neg=concat_query_ids,
            token_type_ids_neg=token_type_ids,
            attention_mask_neg = attention_mask_pos
        )[0]
        final_clf_scores = final_clf_logits[:, 1]
        sorter = torch.argsort(final_clf_scores, -1, descending=True)

        curr_score = final_clf_scores[sorter[0]].item()
        curr_trigger = tokenizer.decode(output_so_far[sorter[0], start_idx:].cpu().tolist())
        trigger_cands.append((curr_score, curr_trigger))
        if curr_score > best_score:
            best_score = curr_score
            best_trigger = curr_trigger
        print("BEST_TRI:", best_trigger, "SCORE:", best_score)
        

        #RANK test
        twithp = best_trigger + " " + raw_passage
        concat_pos = tokenizer([(query, twithp)],  max_length=args.max_seq_len, padding="max_length", add_special_tokens=True,truncation=True, return_tensors='pt').to(device)
        cz_out = model(
            input_ids_pos=concat_pos['input_ids'],
            token_type_ids_pos=concat_pos['token_type_ids'],
            attention_mask_pos=concat_pos['attention_mask'],
            input_ids_neg=concat_pos['input_ids'],
            token_type_ids_neg=concat_pos['token_type_ids'],
            attention_mask_neg=concat_pos['attention_mask'],
        )

        # re-order internal states
        past = lm_model._reorder_cache(present, beam_idx)
        # next
        curr_len = curr_len + 1

    return best_trigger, best_score, trigger_cands

