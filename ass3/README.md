# Intro to NLP 2024-2025, Assignment 3.

**Trying to help people take up their rights!**

**IMPORTANT**: The assignment has two parts. Part 1 is to be submitted **before** Part 2.

## Introduction

This assignment is about retrieval for question answering. To spice things up, we will work in a setup where the technology and models are not as mature as in English, which will highlight multiple challenges of building NLP systems of this kind. You will create an end-to-end system, with multiple components, that has to work "for real". The problem specification is clear, but we leave the solution somewhat open, to allow you to experiment with the different tradeoffs. As usual the report you will produce is a major part of the grade, so invest in it.

The system you will build will be a **question answering (QA) system** that will help people in Israel understand their rights, by answering questions on materials from the **כל זכות** website. (Yes, this means working in Hebrew, where a lot of the challenge comes from). We will create a **"retrieval augmented generation" (RAG)** system, which means that it involves both retrieval an LLMs.

### Retrieval Augmented Generation (RAG)

While we will cover it more formally later in class, the idea of RAG is very simple: when we ask an LLM a question, we also populate the prompt with additional material to help answer the question. In other words, the flow is as follows:

1. User sends a question to the system.
2. The system attempts to retrieve documents from some database, that are likely to contain an answer to the question.
3. The top retrieved document are sent, together with the user's question, to an LLM, in a prompt that says "Consider these documents and answer the following question based on them".
4. The LLM's response is returned to the user, either with or without pointers to the retrieved documents.

Our focus in this assignment will be on items 2 and 3 of this flow.

### Data Collection

Part of the development of an NLP system is to think about the task definition and the data collection. Hence, the first part of this assignment will be dedicated to such data collection.

### Technical Notes:

**Installing packages**: In this assignment and others, we mention various python packages you need to use, as well as instructions on how to install them. For each of them, you are advised/expected to also look at their documentation online. While our installation instructions are to use `pip install` directly, this installs the packages to your main python environment. It is fine, but the current "best practice" in the industry is to install the needed packages for each project in their own "virtual environment". If you want to learn more about virtual environments and tools for managing them, you can google for `venv`, `python uv`, `python poetry` or `pipenv`.

**Using GPU**: The assignment does not require the use of a GPU. While it does involve running neural models, we do not require training and the amount of text is such that can be handled also on a CPU (although it might take a few hours to run). Having said that, GPU can definitely speed some things up. If you want to take advantage of GPU, `google colab` provides some free GPU hours to its users (and of course you are welcome to use whatever other GPU you have access to).

**LLMs**: This assignment involves the use of a large language model (LLM). You can use whatever LLM is convenient for you. Commercial LLMs are available for use through simple web-based APIs with python clients to use them (for example, `pip install openai` for GPT, `pip install anthropic` for Calude). We are sure you could figure it out. Open LLMs can be downloaded and run locally through projects like [ollama](https://ollama.com/), or through API providers such as [together.ai](https://docs.together.ai/docs/chat-overview).
API use is more convenient, often faster, and sometimes higher quality, but also costs a bit of money (though some providers have some free tiers for a small amount of tokens per month). We expect the cost of LLM calls for this project, even through an API, to be very small (we do not require many calls and many tokens), but you can decide if you want to use an API or run a model locally. 

## Part 0: The data (0 points)

This zip file contains the HTML text of over 8000 Hebrew pages from the כל זכות website. 

We also provide the following scripts:

- [src/extract_titles.py]
- [src/basic_extract_text_as_markdown.py]
- [src/get_random_page.py]

The scripts use the `Beautiful Soup` (`pip install bs4`) and the `markdownify` (`pip install markdownify`) python packages for parsing HTML and converting it to markdown format.

Try running them and see that you can make them work. These files are basic, and are not meant to be used as is (though you can if you want) but rather they are meant to be skeletons that you can modify and adapt to your needs in this project.
 
## Part 1:  Dev-set Creation (30 points), submit a week from publication.

In this part, you will create a small annotated dev-set data for evaluating (and for developing) the retrieval part of your project. 

**You will submit this set by the end of the first week** and we will collect all of your individual submissions and release a test-set based on a subset of them. You will have to submit results of your system on the joint test set in your final report.

### What you need to create

The task we are attempting to solve is to answer questions based on the data in the כל זכות corpus, that is, to answer questions that will help people understand the rights available to them, and how to use them.

But how do such questions look like? This is up to you to think of and define, and we ask you to think of (at least) 20 different questions (or queries) to use in your dataset.

The process that we ask you to follow is to sample a random page (using the script provided in part 0), look at it, and try to think of a question that this page answers, and that someone might look for. Once you found a question, record the page, the question, and text from the page that answers this question (the text may come from different parts of the page). Do this until you have **questions from 20 different pages**. (You can have more than one question per page if you want, but you are _required_ to provide 20 pages with at least one question about each).

**What questions to ask?** This is entirely up to you, the main criteria is that the question will be something that someone may realistically want to get an answer for. Note however that the questions are to be part of a search system, that is, while _you_ see the page when answering the question, the user will not see the page, and this will affect the kind of questions they are likely to ask. That is, if the question contains details you are likely not expected to know without reading the page, then it is likely not a good question for this project.

Questions come in many shapes and form, among other things, they can be:
- **Binary**: האם אדם שברשותו דירה זכאי לקבל סיוע בשכר דירה מהמדינה
- **Open with a short answer**: עד כמה חודשים לאחר השחרור זכאי אסיר משוחרר לסיוע בשכר דירה
- **Open with a long answer**: אילו טפסים אלמנים או אלמנות שרוצים לקבל שיקום תעסוקתי צריכים להגיש, ולאן

In all of these, the answer is expected to be on a single page. There are also questions whose answers are likely to be spread across many pages:

- **Aggregate answers**: אילו זכויות מגיעות לאסירים משוחררים

Your questions could also take the form of a query, if you prefer:
* **Query-like**: זכויות אסירים משוחררים דיור

It is OK to include questions from all these kinds, though try to balance between questions that require aggregate answers and ones that don't, leaning towards questions that do not require aggregation over multiple pages. (it is up to you though to decide on the kinds of question you find most interesting / useful).

Of course, you need the question to be also something that you think will be possible to answer. However, try to make things not be "too easy", by not following the language in the page exactly, but to ask a question which differs somewhat in its language than the phrasing of the answer on the page.

To keep the scope of this project to NLP, **avoid asking questions whose answers require understanding a table**.
### What you need to submit

You need to create a dataset of at least 20 queries and their expected answers, each of the 20 queries form a different page. Submit a `.csv` file with 4 columns, in this order:

- The question or query
- The URL of the page containing the answer
- "True" if you expect to all the answer to appear on this page, and "False" if this query may have additional parts of the answer in other pages.
- Text from the page that answers the query/question (can be non-contiguous)

The third column is for "aggregate" questions ("True" for non-aggregate, "False" for aggregate). For aggregate questions, it is sufficient to include a single page from the answer, but try to make it a **central page**, for this question.

Each line should be a single question. You should have exactly 20 lines. It is best to create the `.csv` file by writing in a spreadsheet (such as google-sheets or excel) and exporting to `.csv` ("download as .csv" for google sheets, `save as` for excel).

### Easy or hard?

It is tempting to create questions that are easy to answer using the existing technology, so that the system you develop in part 2 will be able to answer them. However, try to avoid this temptation: we want our questions to be somewhat realistic, so that our evaluation will reflect what will happen with real queries of actual users. To help you avoid the temptation, we will also score your questions based on their hardness: you want at least some of your questions to not be answerable by **other** systems. An ideal question will be answerable by some systems, but not by all of them.

### When to submit

You need to submit this part by @@@.

However, note that the second part is substantially longer in terms of the work you have to do, so we advise that you start part 2 as early as possible.

## Part 2: Retrieval Augmented Generation system (70 points)

In this part we will try and create a system to answer the queries you created in part 1. This may not be easy, but try to answer as many of them as possible. 

This work has four conceptual parts (which interact with each other):
1. Pre-processing and text extraction
2. Indexing the data
3. Query formulation and retrieval
4. RAG-based LLM

Each one of them have multiple choices and options, which are up to you to explore. We will now describe the different parts and give you potential directions and questions to think about for each stage.
### Pre-processing and Text Extraction

We need to answer questions based on text in the כל זכות pages, but these pages come as HTML pages, and contain additional content such as menus, links, etc. We need to process these pages and convert them into textual units which we will then index and retrieve.

We provide you (in part 0) with scripts that extract text from the HTML pages, but they are minimal, and definitely not optimal. You are encouraged to improve these scripts as needed.

In particular, a key question here is "what do we index?" or "what is the basic unit we work with". Do we index entire pages? Or do we index smaller units like paragraphs, or sections, or individual sentences? What is even a paragraph or a section in our case?

Also, if we index something which is smaller than the entire document (like a section) maybe we want to add to it also other items from the page? For example, we may want our unit of indexing to be a pair of "page title + section" or "page title + main paragraph + section". Think about this, it is up to you.

Another option is to index things in multiple levels of granularity: maybe we want to index both the title on its own, _and_ each section on its own, _and_ the entire page, and then the retriever could return any of them.

Another thing to think about is, are there items you want to _not_ index?

The answer to these questions relate a lot to what you will do in the following stages, and you may revise your decisions as you go on. But in any case the first step of your _process_ will be to convert the HTML pages into textual units you can index.

### Indexing

The input to this stage is a set of textual units, and the source (URL or file-name) of each unit. The output is an index that you can query. In response to the query, you should be able to return:

1. The top K most relevant units.
2. The source of each unit.

You may create a single index, or multiple ones. Here are some options and considerations:

**Sparse, lexical, BM25-based indices**

To create a BM25-based index, you can use the [bm25s](https://github.com/xhluca/bm25s) python package (`pip install bm25s[full]`). It is fast and easy to install and use. While it won't work well with very large indices, our current corpus is quite small (~8,000 page, ~80,000 sections, ~800,000 lines), and this package is very adequate.

Sparse lexical indices coupled with BM25-ranking can be very effective, and quite competitive, especially for longer queries (see the "Retrieval" part below), but also for short ones.

Hebrew does pose a challenge though, as we index exact lexical units, which will make us miss things in case of inflections (if we index אמהות and search for אמא, we won't find it) and prefixes (if we index בית and search for לבית, or index לבית and search for בית). You can try (and are encouraged to) to think of ways to try to mitigate this issue, either at indexing time (for example by either stemming, lemmatizing or adding additional words to the text at indexing time (that is, if we index a string אבג we may change the document to be indexed to include also לאבג, ואבג etc)).

It is also possible to enrich the document with "similar words" (for example obtained by word2vec or a masked-language model) to index in addition to the words in the document.

And finally, you may want to filter stop-words before indexing.

**Dense, vector-based indices**

In this approach, each unit is associated with a vector, and during retrieval we encode our query as a vector, and look for the K most similar vectors to our query vector.[^1]

The question, then, is how to obtain the vectors (of course, the assumption is that the vectors encode some sort of semantic similarity). 

Here are a few options:

1. **Combine static word vectors** (each text can be represented as an average or a weighted sum over the individual word vectors). You can find pre-trained Hebrew word vectors [here](https://drive.google.com/drive/folders/1qBgdcXtGjse9Kq7k1wwMzD84HH_Z8aJt).

2. **Combine contextual word-vectors**. Again, each text is represented as an average or a weighted sum of the contextualized vectors of its individual tokens. For Hebrew, you can use existing pre-trained BERT-like models that are available in the Hugging-face `Transformers` library. Specifically you can use [alephbert-base](https://huggingface.co/onlplab/alephbert-base) or  [dictabert](https://huggingface.co/dicta-il/dictabert).[^2] 

3. **Use a pre-trained text embedder**, that is trained specifically for encoding texts as single vectors. While no such model currently exist for Hebrew, there are some "multilingual" ones, that supposedly work with many languages, including Hebrew. In particular, the [sentence-transformers](https://sbert.net/) package has some [pre-trained multilingual models](https://sbert.net/docs/sentence_transformer/pretrained_models.html#multilingual-models), some of which support Hebrew (`he`). LLM APIs of providers like [OpenAI](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings?lang=node) and [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/embeddings) have `embedding` endpoints that will produce results also for Hebrew texts. 

Feel free to experiment! In any case, you need to produce vectors for each unit you want to index, store them in a numpy array, and save the array for future use. In retrieval time, load the array, encode the query using the same method that encoded the vectors in the array, and look for the most similar rows.

Note that the different embedding methods may have a limit on the size of text (number of tokens) that can be encoded in each vector, so you may not be able (and may not want to anyhow) encode complete pages, and have to work with smaller units. 

We note also that while options (2) and (3) work rather well for English texts, in Hebrew the vector representations are sub-optimal. This is a challenge we need to work with. Training dedicated Hebrew text embedders is very much worthwhile, but is out of scope for this assignment.[^3] 

_Some technical notes_:

In both BM25 and Dense indexing, the index usually takes a query and returns a document ID, and you store the document-id-to-document mapping separately. For example, in the dense case, your index is essentially a list of N vectors, and the retrieval returns the positions of the K most similar vectors to your query vector. You need to store an additional list of N documents, whose indices align to those of the corresponding vectors. 

When embedding texts into vectors, collect the embedded vectors into a python list (each vector is an item in the list, and can be either a list of floats or a numpy vector). Then, you can convert the list into a numpy array with `arr = numpy.array(our_list)` and save it with `numpy.save(open("filename.npy", "wb"), arr)`. (Load with `numpy.load`). Searching the array for similar vectors to a vector Q can done using:

```python
scored = arr @ Q
topK = numpy.argsort(-scored)[:K]
```

If you have a `numpy` array of `docs` documents corresponding to the vectors in `arr`, you can then get the documents with `docs[topK]`.

Embedding the text units may take some time (maybe up to several hours) on the CPU. I'd recommend saving intermediate results (say, save every K items that you encode). If working on a GPU, it will be much faster to encode a batch of K documents at a time, rather than one by one.

To convert a pytorch tensor to a numpy array/vector, use: `arr = pt.detach().cpu().numpy`.

**Enriched Documents**

You may index, in addition to (or even instead of) the document text, also other texts. For example, you may ask an LLM to ask questions about the page, and index the questions. You may ask an LLM to summarize a page, and index the summary. You may translate the text and index the translation. These indices could be either sparse or dense, and the same or separate from your main indices. This is optional of course, but something to think of.
### Retrieval

Now that you have an index, we should perform retrieval to obtain the relevant texts for our query. Retrieval can be very simple: just use the user's query text as your query, and obtain the results. However, it can also be more elaborate: for example, we can:

- use _query enrichment_ to add words to our query
- ask an LLM to translate the question into a good query (or into multiple queries)
- use a method like HyDE (discussed in class, but briefly: use an LLM to generate an answer, and then use the answer as the query)
- make multiple rounds of retrieval, where the later rounds depend on the results of the earlier rounds (again, maybe an LLM could help with this)
- etc.

You can be creative here, and it can improve the system.

_Reranking_
Another possibility to do in the retrieval stage is _reranking_: if you want to feed $k$ items to the prompt, retrieve $m \gt k$ items, and then choose the best $k$ of the $m$ based on a stronger model, that sees the query and the item together.

### End-to-end RAG System

Finally, we got to the end-to-end system. Wrap it all together in a system that:

- Gets a query from the user.
- Retrieve relevant documents from the index (or several indices) that you previously created.
- Send a prompt to an LLM that combines the question and retrieved documents[^4], and instructions to answer the question based on the text in the prompt. Then, return the answer to the user.

This is mostly technical, but tying everything together is important, and the phrasing of the prompt does matter. If you want to go overboard, it will be nicer if the prompt can not only answer the question, but also cite the document (or documents) from which it found the answer, and provide links to them.

_How should the inputs and outputs look?_

We do not ask you to submit code we can run, so do whatever is convenient to you. Something that is easy to do is to, for example, expose the system as a function called "Query" that takes in a string and returns the result. You can then call it several times in a python notebook. Somewhat more elaborate is to have a script that takes the query as a commandline parameter, and another option is to create a quick UI with a system like [streamlit](https://streamlit.io/) (a nice to know system). But really, this is up to you. We will not see this part. It is helpful to have something convenient to work with and look at the results when you develop your system, though.

### Evaluation

We ask you to perform two kinds of evaluation:

1. **End-to-end evaluation**, in which you go from query to final answer. This will have to be done manually. Choose 10-20 queries and manually evaluate their answers. Beyond accuracy, see if you can find common trends, or cases where the retrieval found the correct document but the overall system produced a wrong answer, or the other way around: cases where the retrieval failed but the system overall produced an adequate answer.

2. **Retrieval Evaluation**, here, we evaluate the system on their ability to find the correct page (ignoring the RAG part). Use two metrics: The first metric is _recall@k_ which measures how many times the correct page was within the top-k retrieved documents (after reranking, if you do them). Use k=5 and k=20. The second metric is [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank), which looks for the position in which you ranked the correct document for the query.

Your report should include both the end-to-end evaluation, and also the retrieval evaluation. For the retrieval evaluation, you should report numbers for two sets of queries: 
- Your own dev-set you created in part 1.
- The test-set we release which is based on queries from all the teams.

We also ask you to submit a file called `ranks.txt` in which each line corresponds to a single query from the test-set, and contains two numbers separated by space: `QUERY_ID RANK` where `QUERY_ID` is the ID of the query, and `RANK` is the position in which the relevant page was returned by your retriever (0 for first position, 1 for second position, etc).

## What to submit

**For part 1**, submit a file called `questions.csv` with the queries you created, as well as a file called `README.md` with (1) names and ids of team members, and (b) a brief description of your queries and the process of creating them, and any additional thoughts you think are relevant.

**For part 2**, submit 

(1) The predictions file `ranks.txt`

(2) Code for your pre-processing, indexing, retrieval and RAG parts (if you use API keys, be sure to remove them from the code before submitting). Include also a README.md file the describes which files belong to which part. 

We will not try to run the code, but we will look at it.

(3) _a detailed report_ in a file called _report.pdf_ which should include:
- Your names
- A description of your system and the choices you made. We should be able to understand how your system works in all the different parts, and especially the key choices and key points that are required to replicate it. If something worked particularly well, specify it.
- The evaluation results you obtained.
- If you experimented with different methods and ideas, it is good to describe not only what worked, but also what didn't work.
- Any additional thoughts you had based on the assignment, or ideas you may have.

The your grade will be based mostly on the report, so invest time in it!

## Good luck!

# Footnotes

[^1]: This is very common, and there are systems like FAISS or Vespa that allow to perform efficient k-most-similar searches on large vector collections, with billions of vectors. However, for our small corpus, we can just keep all the vectors as a `numpy` array, and search using a simple dot-product operation.
[^2]: In both cases, make sure that you use the representation vectors and not the vocabulary-prediction vectors, which are much larger (use `BertModel` and not `AutoModelForMaskedLM`). 
[^3]: Unless you really really want to. Note that the `sentence-transformers` package provides training code.
[^4]: Note that you can include exactly what you retrieved, or something a bit different. For example, if you retrieved a piece of text that resulted from page X, you may include in the prompt also additional information from page X, if you think it is useful.

