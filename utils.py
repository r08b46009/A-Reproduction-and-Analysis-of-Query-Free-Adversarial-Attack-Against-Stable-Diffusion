import torch
import copy
import random
from torch.nn import functional as F
import numpy as np
from PIL import Image

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)


def _get_device(text_encoder):
    return next(text_encoder.parameters()).device


def _build_causal_attention_mask_manual(bsz, seq_len, dtype, device):
    mask = torch.full(
        (bsz, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask.triu_(1)
    return mask.unsqueeze(1)


def get_text_embeds_without_uncond(prompt, tokenizer, text_encoder):
    device = _get_device(text_encoder)

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_input.input_ids.to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(input_ids)[0]

    return text_embeddings


def get_char_table():
    char_table = [
        "·", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "=", "-", "*",
        "+", ".", "<", ">", "?", ",", "'", ";", ":", "|", "\\", "/"
    ]
    for i in range(ord("A"), ord("Z") + 1):
        char_table.append(chr(i))
    for i in range(0, 10):
        char_table.append(str(i))
    return char_table


def cos_mask(a, b, mask):
    return cos(a * mask, b * mask)


# greedy algorithm
def search_min_char(sentence_embading, sentence, char_list, k, mask=None, tokenizer=None, text_encoder=None):
    modify_sentence = copy.deepcopy(sentence)
    res = ""
    min_cos = 1
    min_char = ""

    for c in char_list:
        change_sentence = list(copy.deepcopy(modify_sentence))
        change_sentence[k] = c
        change_sentence = "".join(change_sentence)

        change_embading = get_text_embeds_without_uncond([change_sentence], tokenizer, text_encoder)

        if mask is None:
            temp_cos = cos(sentence_embading.view(-1), change_embading.view(-1))
        else:
            temp_cos = cos_mask(sentence_embading.view(-1), change_embading.view(-1), mask)

        if temp_cos < min_cos:
            res = change_sentence
            min_cos = temp_cos
            min_char = c

    modify_sentence = res
    return modify_sentence


def search_min_sentence_iteration(sentence, char_list, length, iter_times, mask=None, random_choice=False, tokenizer=None, text_encoder=None):
    sentence_embedding = get_text_embeds_without_uncond([sentence], tokenizer, text_encoder)

    if random_choice:
        first_c = random.choice(char_list)
        modify_sentence = copy.deepcopy(sentence) + " " + first_c
        length -= 1
    else:
        modify_sentence = copy.deepcopy(sentence) + " "

    for _ in range(length):
        modify_sentence += " "
        modify_sentence = search_min_char(
            sentence_embedding,
            modify_sentence,
            char_list,
            -1,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    for _ in range(iter_times):
        for k in range(length, 0, -1):
            modify_sentence = search_min_char(
                sentence_embedding,
                modify_sentence,
                char_list,
                -k,
                mask,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

    return modify_sentence


# genetic algorithm
def get_generation(string1, string2, char_list, cross_loc=None, variation_loc=None):
    if len(string1) != len(string2):
        print("length of string1 and string2 should be the same")
        return None
    string1, string2 = cross_generation(string1, string2, cross_loc=cross_loc)
    string1, string2 = vari_generation(string1, string2, char_list, vari_loc=variation_loc)
    return string1, string2


def cross_generation(string1, string2, cross_loc=None):
    if cross_loc is None:
        cross_loc = random.randint(1, len(string1) - 1)

    string1_seg1 = string1[0:cross_loc]
    string2_seg1 = string2[0:cross_loc]
    string1_list = list(string1)
    string2_list = list(string2)

    for i in range(len(string1_seg1)):
        string1_list[i] = string2_seg1[i]
        string2_list[i] = string1_seg1[i]

    string1 = "".join(string1_list)
    string2 = "".join(string2_list)
    return string1, string2


def vari_generation(string1, string2, char_list, vari_loc=None):
    if vari_loc is None:
        vari_loc = random.randint(0, len(string1) - 1)

    vari_char = random.randint(0, len(char_list) - 1)
    string1_list = list(string1)
    string2_list = list(string2)
    string1_list[vari_loc] = char_list[vari_char]
    string2_list[vari_loc] = char_list[vari_char]
    string1 = "".join(string1_list)
    string2 = "".join(string2_list)
    return string1, string2


def genetic(sentence, char_list, length, generation_num=50, generateion_scale=20, mask=None, tokenizer=None, text_encoder=None):
    generation_list = init_pool(char_list, length)
    score_list = {}

    for _ in range(generation_num):
        pool = []
        for candidate in generation_list:
            mate = random.choice(generation_list)
            g1, g2 = get_generation(candidate, mate, char_list)
            pool.append(g1)
            pool.append(g2)

        generation_list = select(
            sentence,
            pool,
            generateion_scale,
            score_list=score_list,
            mask=mask,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    res = sorted(score_list.items(), key=lambda x: x[1], reverse=False)[0:5]
    return res


def select(sentence, pool, generateion_scale, mask=None, score_list=None, tokenizer=None, text_encoder=None):
    text_embedding = get_text_embeds_without_uncond([sentence], tokenizer, text_encoder)
    pool_score = []

    if score_list is None:
        score_list = {}

    for candidate in pool:
        if candidate in score_list:
            temp_score = score_list[candidate]
            pool_score.append((temp_score, candidate))
            continue

        candidate_text = sentence + " " + candidate
        if mask is None:
            temp_score = cos_embedding_text(text_embedding, candidate_text, tokenizer=tokenizer, text_encoder=text_encoder)
        else:
            temp_score = cos_embedding_text(text_embedding, candidate_text, mask, tokenizer=tokenizer, text_encoder=text_encoder)

        score_list[candidate] = temp_score
        pool_score.append((temp_score, candidate))

    sorted_pool = sorted(pool_score)
    selected_generation = [x[1] for x in sorted_pool]
    return selected_generation[0:generateion_scale]


def cos_embedding_text(embading, text, mask=None, tokenizer=None, text_encoder=None):
    change_embading = get_text_embeds_without_uncond([text], tokenizer, text_encoder)
    local_cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    if mask is None:
        return local_cos(embading.view(-1), change_embading.view(-1)).item()
    else:
        return local_cos(embading.view(-1) * mask, change_embading.view(-1) * mask).item()


def init_pool(char_list, length, num=10):
    pool = []
    for _ in range(num):
        pool.append("".join(random.sample(char_list, length)))
    return pool


# PGD
def get_clip_embedding(prompt, tokenizer=None, text_encoder=None):
    device = _get_device(text_encoder)

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_input.input_ids
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1]).to(device)

    with torch.no_grad():
        txt_embed = text_encoder.text_model.embeddings(input_ids=input_ids)

    return txt_embed, input_shape


def forward_embedding(hidden_states, input_shape, model=None, tokenizer=None, text_encoder=None):
    output_attentions = text_encoder.text_model.config.output_attentions
    output_hidden_states = text_encoder.text_model.config.output_hidden_states
    bsz, seq_len = input_shape
    return_dict = text_encoder.text_model.config.use_return_dict

    causal_attention_mask = _build_causal_attention_mask_manual(
        bsz, seq_len, hidden_states.dtype, hidden_states.device
    )
    attention_mask = None

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
    return last_hidden_state


def forward_embedding_no_grad(hidden_states, input_shape, model=None, tokenizer=None, text_encoder=None):
    with torch.no_grad():
        output_attentions = text_encoder.text_model.config.output_attentions
        output_hidden_states = text_encoder.text_model.config.output_hidden_states
        bsz, seq_len = input_shape
        return_dict = text_encoder.text_model.config.use_return_dict

        causal_attention_mask = _build_causal_attention_mask_manual(
            bsz, seq_len, hidden_states.dtype, hidden_states.device
        )
        attention_mask = None

        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    return last_hidden_state


class PGDattack:
    def __init__(self):
        self.stdste = True

    def project_u_tensor(self, u_tensor, site_mask, sub_mask):
        skip = site_mask == 0
        subword_opt = sub_mask != 0

        for i in range(u_tensor.size(0)):
            if skip[i]:
                continue
            u_tensor[i][subword_opt[i]] = self.bisection_u(u_tensor[i][subword_opt[i]], eps=1)
            assert torch.abs(torch.sum(u_tensor[i][subword_opt[i]]) - 1) <= 1e-3

        return u_tensor

    def bisection_u(self, a, eps, xi=1e-5, ub=1):
        pa = torch.clip(a, 0, ub)

        if np.abs(torch.sum(pa).item() - eps) <= xi:
            upper_S_update = pa
        else:
            mu_l = torch.min(a - 1).item()
            mu_u = torch.max(a).item()

            while np.abs(mu_u - mu_l) > xi:
                mu_a = (mu_u + mu_l) / 2
                gu = torch.sum(torch.clip(a - mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a - mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a - mu_u, 0, ub)) - eps

                if gu == 0:
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break

                if gu * gu_l < 0:
                    mu_u = mu_a
                elif gu * gu_u < 0:
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()

            upper_S_update = torch.clip(a - mu_a, 0, ub)

        return upper_S_update

    def estimate_u_tensor(self, u_tensor, site_mask):
        if self.stdste:
            return F.gumbel_softmax(u_tensor, tau=0.5, hard=True, dim=-1)

        for i in range(u_tensor.size(0)):
            if site_mask[i] == 0:
                continue
            u_tensor[i] = STERandSelect.apply(u_tensor[i])

        return u_tensor


class STERandSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        prob = input.cpu().detach().numpy()
        prob = prob / np.sum(prob)
        substitute_idx = np.random.choice(input.size(0), p=prob)
        new_vector = torch.zeros_like(input).to(input.device)
        new_vector[substitute_idx] = 1
        return new_vector

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def craft_candidate_embed(char_list, tokenizer=None, text_encoder=None):
    prompt_char = ""
    position_length = 75
    candidate_embedding = []

    for i in char_list:
        prompt_char = ""
        for _ in range(position_length):
            prompt_char += i
            prompt_char += " "

        char_posit_embed = get_clip_embedding(prompt_char, tokenizer=tokenizer, text_encoder=text_encoder)[0]
        candidate_embedding.append(char_posit_embed)

    candidate_embedding = torch.stack(candidate_embedding, dim=1)
    candidate_embedding = candidate_embedding.permute(0, 2, 1, 3)
    return candidate_embedding


def train(init_per_sample, sentence, len_prompt, char_list, model, iter_num=100, eta_u=1, mask=None, tokenizer=None, text_encoder=None):
    device = _get_device(text_encoder)

    num_word = len(sentence.split(" "))
    seq_len = 77
    neighbor_num = len(char_list)

    u_tensor = torch.zeros(
        [init_per_sample, seq_len, neighbor_num],
        dtype=torch.double,
        device=device,
    ).fill_(1 / neighbor_num)

    fixed_z = torch.zeros(init_per_sample, seq_len, device=device)
    fixed_z[0][num_word + 1:num_word + len_prompt + 1] = 1
    u_tensor.requires_grad = True

    candidate_embed = craft_candidate_embed(char_list, tokenizer=tokenizer, text_encoder=text_encoder)
    candidate_embed = candidate_embed.to(device)

    orig_embeddings, input_shape = get_clip_embedding(sentence, tokenizer=tokenizer, text_encoder=text_encoder)
    orig_embeddings = orig_embeddings.to(device)

    orig_output = get_text_embeds_without_uncond([sentence], tokenizer=tokenizer, text_encoder=text_encoder)
    orig_output = orig_output.to(device)

    if mask is not None:
        mask = mask.to(device)

    model.train()
    loss_list = []
    single_loss_list = []
    batch_size = 100
    sign = True
    simplex = False
    bool_max = True
    n = 5
    pgd = PGDattack()
    max_tensor = None

    for _ in range(iter_num):
        topn_vector = []

        for _ in range(batch_size):
            discrete_u = pgd.estimate_u_tensor(u_tensor, fixed_z.view(-1))
            discrete_u = discrete_u.view(init_per_sample, seq_len, neighbor_num, 1).to(device)

            subword_embeddings = candidate_embed.view(1, seq_len, neighbor_num, -1)
            discrete_z = fixed_z.view(init_per_sample, seq_len, 1).to(device)
            orig_embeddings_reshaped = orig_embeddings.view(1, seq_len, -1)

            new_embeddings = (1 - discrete_z) * orig_embeddings_reshaped + discrete_z * torch.sum(
                discrete_u * subword_embeddings, dim=2
            )
            new_embeddings = new_embeddings.float()

            new_output = forward_embedding_no_grad(
                new_embeddings,
                [1, 77],
                model,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

            if mask is not None:
                temp_loss = 1 / cos_mask(new_output.view(-1), orig_output.view(-1), mask)
            else:
                temp_loss = 1 / cos(new_output.view(-1), orig_output.view(-1))

            single_loss_list.append(temp_loss.item())

            if len(topn_vector) < n:
                topn_vector.append((temp_loss.item(), discrete_u))
                topn_vector = sorted(topn_vector, reverse=True)
            else:
                if temp_loss.item() >= topn_vector[-1][0]:
                    topn_vector.append((temp_loss.item(), discrete_u))
                    topn_vector = sorted(topn_vector, reverse=True)[:n]

            if temp_loss.item() >= max(single_loss_list):
                max_tensor = discrete_u

        total_loss = 0

        for k in range(len(topn_vector)):
            max_vector = topn_vector[k][1]
            subword_embeddings = candidate_embed.view(1, seq_len, neighbor_num, -1)
            discrete_z = fixed_z.view(init_per_sample, seq_len, 1).to(device)
            orig_embeddings_reshaped = orig_embeddings.view(1, seq_len, -1)

            new_embeddings = (1 - discrete_z) * orig_embeddings_reshaped + discrete_z * torch.sum(
                max_vector * subword_embeddings, dim=2
            )
            new_embeddings = new_embeddings.float()

            new_output = forward_embedding(
                new_embeddings,
                [1, 77],
                model,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

            if mask is not None:
                loss = 1 / cos_mask(new_output.view(-1), orig_output.view(-1), mask)
            else:
                loss = 1 / cos(new_output.view(-1), orig_output.view(-1))

            total_loss += loss

        total_loss = total_loss / n
        loss_list.append(total_loss.item())
        total_loss.backward(retain_graph=True)

        lr = eta_u / np.sqrt(iter_num)
        u_grad = u_tensor.grad

        if sign:
            u_grad = torch.sign(u_grad)

        u_clone = u_tensor.detach().clone()
        u_update = lr * u_grad

        if bool_max:
            u_tensor_opt = u_clone + u_update
        else:
            u_tensor_opt = u_clone - u_update

        sub_mask = torch.ones_like(u_tensor_opt)
        u_tensor_shape = u_tensor_opt.shape

        if simplex:
            u_tensor_opt = pgd.project_u_tensor(u_tensor_opt[0], fixed_z.view(-1), sub_mask[0])

        u_tensor_opt = u_tensor_opt.view(u_tensor_shape)
        u_tensor.data = u_tensor_opt

        if u_tensor.grad is not None:
            u_tensor.grad.zero_()

    if max_tensor is None:
        raise RuntimeError("PGD did not produce any candidate.")

    res_list = max_tensor.view(77, -1)[num_word + 1:num_word + len_prompt + 1].argmax(dim=1)
    return max_tensor, loss_list, "".join([char_list[x] for x in res_list]), max(single_loss_list)


def object_key(sentence_list, object_word, thres=10, tokenizer=None, text_encoder=None):
    diff_list = []
    total_diff = 0

    for i in sentence_list:
        sen_embed = get_text_embeds_without_uncond(i, tokenizer=tokenizer, text_encoder=text_encoder)
        crafted_embed = get_text_embeds_without_uncond(i.replace(object_word, ""), tokenizer=tokenizer, text_encoder=text_encoder)
        diff_list.append(crafted_embed - sen_embed)
        total_diff += crafted_embed - sen_embed

    average_diff = total_diff / len(diff_list)

    total_sign = 0
    for vec in diff_list:
        vec = vec.clone()
        vec[vec > 0] = 1
        vec[vec < 0] = -1
        total_sign += vec

    total_sign[abs(total_sign) <= thres] = 0
    total_sign[abs(total_sign) > thres] = 1
    print("Ratio of mask", total_sign[total_sign > 0].shape[0] / total_sign.view(-1).shape[0])
    return total_sign


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
