def count_words(text):
    return len(text.split())

def summarize(record, llm, word_count_target=None, otherinstructions=None):
    if word_count_target is not None:
      messages = [
        ("system", """
        You are a helpful summary chatbot.  Summarize the content provided by the user in under {word_count_target} words.
        Do not exceed {word_count_target} words when generating a summary.
        """),
        ("human", f"{record['text']}"),
      ]
    else:
      messages = [
        ("system", """
        You are a helpful summary chatbot.  Summarize the content provided by the user. {otherinstructions}
        """),
        ("human", f"{record['text']}"),
      ]
    # TODO should I do some error handling here?
    ai_summary = llm.invoke(messages).content
    return {
        'text_word_count': count_words(record['text']),
        'summary_word_count': count_words(record['summary']),
        'ai_summary': ai_summary,
        'ai_summary_word_count': count_words(ai_summary)
    }
