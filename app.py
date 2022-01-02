from transformers import pipeline
import gradio as gr
from gradio.mix import Parallel, Series
io1 = gr.Interface.load('huggingface/sshleifer/distilbart-cnn-12-6')
io2 = gr.Interface.load("huggingface/facebook/bart-large-cnn")
io3 = gr.Interface.load("huggingface/google/pegasus-xsum")                  
desc =  "Let Hugging Face models summarize texts for you. Note: Shorter articles generate faster summaries. This summarizer uses bart-large-cnn model by Facebook, pegasus by Google and distilbart-cnn-12-6 by Sshleifer. You can compare these models against each other on their performances. Sample Text input is provided!"
        
x = """ What's A Lawyer Now? Simply put… there is a tremendous manifest and latent need for just about ALL legal services. There are solid interrelated sociological and structural reasons for this including considerable societal divisiveness, meaningful changes in laws and regulations, and fast-paced disruptive technological innovations. At the same time, there are psychological factors that strongly prompt the need for various legal services such as hubris, arrogance, and Machiavellianism. The opportunities, across a wide spectrum of law firm practice areas, have probably never been greater. Although there is a tremendous amount of untapped potential for legal services, there is one major obstacle to opening the spigot – lawyers. From solo practices to mega-international law firms, many lawyers because of their inherent inclinations (e.g., risk aversion) reinforced by their education and firm experience are not going to take advantage of the incredible latent demand for legal services. As commoditization is rampant in the legal profession, the path to success is not just having “excellent knowledge of the law.” Being technical proficient is table stakes. Unfortunately, a large percentage of lawyers equate legal competence with the success of their practice, and the great majority is proven wrong. What is also required of lawyers at all levels, in order to truly excel in today’s legal environment, is a touch of entrepreneurialism coupled with some business savvy. The opportunities for lawyers are most everywhere from inside their own book of business to the clients of other lawyers in their firms to the many other types of professionals they know or can fairly easily get to know. The complication is that when it comes to the business development side of legal work, few lawyers have the expertise to create a steady stream of new work for their practices or their firms. Unless lawyers adopt these best practices, it is unlikely that they will be able to greatly benefit from all the tremendous pent up demand that exists for legal services. Conversely, for those lawyers who take a proactive and systemic approach to business development, their practices could easily grow exponentially.
"""
y = '''What is Text Summarization?
Text summarization is an important NLP task, which has several applications. The two broad categories of approaches to text summarization are extraction and abstraction. Extractive methods select a subset of existing words, phrases, or sentences in the original text to form a summary. In contrast, abstractive methods first build an internal semantic representation and then use natural language generation techniques to create a summary. Such a summary might contain words that are not explicitly present in the original document. Most text summarization systems are based on some form of extractive summarization.
In general, topic identification, interpretation, summary generation, and evaluation of the generated summary are the key challenges in text summarization. The critical tasks in extraction-based summarization are identifying key phrases in the document and using them to select sentences in the document for inclusion in the summary. In contrast, abstraction-based methods paraphrase sections of the source document.
All extraction-based summarizers perform the following three relatively independent tasks (Nenkova and McKeown, 2011, 2012): (a) capturing key aspects of text and storing as an intermediate representation, (b) scoring sentences in the text based on that representation, (c) and composing a summary by selecting several sentences.'''
z = '''Machine Learning Technology Trends To Impact Business in 2022
In this article, we will discuss the latest innovations in machine learning technology in 2021 from our perspective as a machine learning software development company. We’ll go over 9 trends and explain how the latest innovations in machine learning technologies can benefit you and your business in 2022.
1. No-Code Machine Learning
2. TinyML
3. AutoML
4. Machine Learning Operationalization Management
5. Full-stack Deep Learning
6. Generative Adversarial Networks
7. Unsupervised ML
8. Reinforcement Learning
 '''
sample = [[y],[x],[z]]
iface = Parallel(io1, io2, io3,
                 theme='huggingface', 
                 title= 'Hugging Face Text Summarizer', 
                 description = desc,
                 examples=sample, #replace "sample" with directory to let gradio scan through those files and give you the text
                 inputs = gr.inputs.Textbox(lines = 10, label="Text"))
iface.launch(inline = False)
