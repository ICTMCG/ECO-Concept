import numpy as np
from IPython.core.display import display, HTML

colors = {
    0: "rgba(9, 221, 55, ",
    1: "rgba(9, 221, 161, ",
    2: "rgba(9, 175, 221, ",
    3: "rgba(221, 9, 34, ",
    4: "rgba(221, 9, 140, ",
    5: "rgba(221, 90, 9, ",
    6: "rgba(255, 102, 122, ",
    7: "rgba(179, 174, 54, ",
    8: "rgba(45, 136, 179, ",
    9: "rgba(255, 222, 173, ",
    10: "rgba(100, 149, 237, ",
    11: "rgba(238, 230, 133, ",
    12: "rgba(60, 179, 113, ",
    13: "rgba(205, 155, 155, ",
    14: "rgba(221, 160, 221, ",
    15: "rgba(205, 181, 205, ",
    16: "rgba(188, 210, 238, ",
    17: "rgba(154, 192, 205, ",
    18: "rgba(32, 178, 170, ",
    19: "rgba(100, 149, 237, ",
}


def vis_concepts(
        contents,
        slot_attention,
        contributions,
        summary_list,
        y_true,
        y_pred,
        cases_ids,
        tokenizer,
        vis_threshold=0.3,
        labels_type=["Negative", "Positive"]
):
    batch_size, num_concepts, max_len = slot_attention.shape

    html_content = "<h2>" + " Concepts" + "</h2>"

    for concept_id in range(num_concepts):
        concept_logits = np.sum(np.array(slot_attention), axis=-1)[:, concept_id]
        concept_attention = slot_attention[:, concept_id, :]

        for case_id in cases_ids[concept_id]:
            tokens = tokenizer.tokenize(contents[case_id])
            clean_tokens = [tokenizer.convert_tokens_to_string(token) for token in tokens]

            titles = ['Model', 'Concept', 'Summary', 'Case', 'Prediction', 'Label', 'Concept Intensity',
                      'Concept Contribution']
            values = ['ECO-Concept', str(concept_id), summary_list[concept_id], str(case_id),
                      map_label(y_pred[case_id], labels_type), map_label(y_true[case_id], labels_type),
                      str(concept_logits[case_id]), str(contributions[case_id][:, concept_id])]
            html_content += get_table(titles, [values])
            html_content += '<br><br>'

            html_text = vis_case(concept_id, clean_tokens, concept_attention[case_id][1:], vis_threshold)
            html_content += "<div style='display: grid; grid-template-columns: 1fr 1fr;'>" + html_text + "</div>"
            html_content += '<br><br>'

    display(HTML(html_content))

    return html_content


def vis_case(concept_id, tokens, activations, vis_threshold):
    phi_html = []
    for i in range(min(len(tokens), len(activations))):
        if float_check((activations[i])):
            if float(activations[i]) > vis_threshold:
                phi_html.append(
                    f'<span style="background-color: {colors[concept_id]} {float(activations[i])}); padding: 1px 5px; border: solid 3px ; border-color: {colors[concept_id]} 1); #EFEFEF">{tokens[i]}</span>')
            else:
                phi_html.append(
                    f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{tokens[i]}</span>')
        else:
            phi_html.append(
                f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{tokens[i]}</span>')
    html_content = "<div style='display: flex; width: 1000px; flex-wrap: wrap'>" + " ".join(phi_html) + " </div>"
    return html_content


def get_table(titles, all_values):
    table_html = "<table border='1'><tr>"
    for t in titles:
        table_html += '<th>' + t + '</th>'
    table_html += '</tr>'

    for values in all_values:
        table_html += '<tr>'
        for v in values:
            table_html += '<td style="font-size: 16px;">' + v + '</td>'
        table_html += '</tr>'
    table_html += '</table>'
    return table_html


def map_label(arr, labels_type=['Negative', 'Positive']):
    l = np.where(arr == np.max(arr))[0][0]
    return labels_type[l]


def float_check(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
