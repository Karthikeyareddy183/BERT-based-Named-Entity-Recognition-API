import html
import logging
import streamlit as st
from ner_service import NERService, ModelUnavailable

ENTITY_STYLES = {
    'PER': ('Person', '#fde2e4', '#9d174d'),
    'ORG': ('Organization', '#dbeafe', '#1e40af'),
    'LOC': ('Location', '#dcfce7', '#166534'),
    'MISC': ('Miscellaneous', '#fef3c7', '#92400e'),
}

EXAMPLES = {
    'Tech news': 'Sundar Pichai, the CEO of Google, met Narendra Modi in New Delhi '
                 'to discuss the AI Summit.',
    'Sports': 'Lionel Messi scored twice as Argentina beat France in the World Cup final '
              'at Lusail Stadium in Qatar.',
    'Business': 'Microsoft announced a partnership with OpenAI, while Satya Nadella '
                'spoke to investors in Seattle.',
    'Simple': 'John works at Google in New York.',
}

CSS = """
<style>
.block-container {padding-top: 2.5rem; max-width: 1100px;}
.ner-text {font-size: 1.15rem; line-height: 2.4; padding: 1.2rem 1.4rem;
    border: 1px solid rgba(128,128,128,.25); border-radius: 12px; margin-bottom: 1.5rem;}
.ner-ent {padding: .2rem .5rem; border-radius: 8px; font-weight: 600; white-space: nowrap;}
.ner-tag {font-size: .68rem; font-weight: 700; margin-left: .35rem; letter-spacing: .04em; opacity: .85;}
.ner-legend {display: inline-block; margin: 0 .4rem .4rem 0;}
</style>
"""


@st.cache_resource
def load_model():
    return NERService()


def badge(text, label):
    _, background, color = ENTITY_STYLES.get(label, ('', '#e5e7eb', '#111827'))
    return (f'<span class="ner-ent" style="background:{background};color:{color}">'
            f'{html.escape(text)}<span class="ner-tag">{label}</span></span>')


def highlight(text, entities):
    parts, cursor = [], 0
    for entity in sorted(entities, key=lambda e: e['start']):
        parts.append(html.escape(text[cursor:entity['start']]))
        parts.append(badge(text[entity['start']:entity['end']], entity['entity']))
        cursor = entity['end']
    parts.append(html.escape(text[cursor:]))
    return f'<div class="ner-text">{"".join(parts)}</div>'


def sidebar():
    with st.sidebar:
        st.header('About')
        st.write('A **BERT** model fine-tuned on the **CoNLL-2003** dataset to find '
                 'people, organizations, locations and other named entities in text.')
        col1, col2 = st.columns(2)
        col1.metric('F1 score', '94.8%')
        col2.metric('Accuracy', '99.1%')
        st.subheader('Entity types')
        st.markdown(''.join(f'<div class="ner-legend">{badge(name, label)}</div>'
                            for label, (name, _, _) in ENTITY_STYLES.items()),
                    unsafe_allow_html=True)
        st.caption('Built with Hugging Face Transformers, PyTorch and Streamlit.')


def results(text, entities):
    st.subheader('Highlighted text')
    st.markdown(highlight(text, entities), unsafe_allow_html=True)
    if not entities:
        st.info('No named entities found in this text.')
        return

    st.subheader('Summary')
    columns = st.columns(len(ENTITY_STYLES))
    for column, (label, (name, _, _)) in zip(columns, ENTITY_STYLES.items()):
        column.metric(name, sum(entity['entity'] == label for entity in entities))

    st.subheader('Entities found')
    st.dataframe(
        [{'Entity': e['word'], 'Type': ENTITY_STYLES.get(e['entity'], (e['entity'],))[0],
          'Label': e['entity'], 'Confidence': round(e['score'] * 100, 1)} for e in entities],
        hide_index=True, width='stretch',
        column_config={'Confidence': st.column_config.ProgressColumn(
            'Confidence', format='%.1f%%', min_value=0, max_value=100)})


def main():
    st.set_page_config(page_title='BERT NER', page_icon='🔎', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()

    st.title('🔎 Named Entity Recognition')
    st.write('Paste any text and the model will find **people**, **organizations**, '
             '**locations** and **other entities** in it.')

    example = st.selectbox('Try an example', list(EXAMPLES))
    text = st.text_area('Your text', EXAMPLES[example], height=130, max_chars=10000)

    if st.button('Find entities', type='primary'):
        try:
            with st.spinner('Analyzing...'):
                st.session_state.result = (text, load_model().predict_detailed(text))
        except ValueError as error:
            st.session_state.pop('result', None)
            st.warning(str(error))
        except ModelUnavailable:
            st.session_state.pop('result', None)
            st.error('NER model unavailable. Check NER_MODEL_PATH and model artifacts.')
        except Exception:
            st.session_state.pop('result', None)
            logging.exception('NER inference failed')
            st.error('NER inference failed.')

    if 'result' in st.session_state:
        st.divider()
        results(*st.session_state.result)


if __name__ == '__main__':
    main()
