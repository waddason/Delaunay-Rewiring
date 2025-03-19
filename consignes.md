# ML with graphs - project

## Defense

The instructions are the following:
- You must **present the paper in 10 minutes + 5 minutes for questions**. I’ll need to be strict on the time constraint.
- The goal is not to present everything that is in the paper. You should **focus on presenting the key technical novelty introduced in the paper and explain how it advances the state of the art**. You can present the key experimental results but you should not try to cover all the experiments presented in the paper. You will be evaluated mostly on your **capacity to identify the most relevant information to summarize the paper**.
- Additionally, please try to **reproduce at least one of the experiments and compare your results** with what’s reported in the paper. If you’re unable to reproduce the results, you should explain why you were not able to run at least one of the experiments (Out-Of-Memory? High computational complexity? etc). 
- We recommend preparing a few slides, please do so in PDF format. The slides should be **submitted by March 24** to a link we’ll provide later. This is a strict deadline no matter if you’re presenting on March 25th or April 1st.

## Report
**The report is due on April 8th**. It's limited to 4 pages of content (excluding references) using the ICML template (https://www.overleaf.com/latex/templates/icml2025-template/dhxrkcgkvnkt).

This report is divided into two distinct sections. The first section is dedicated to **summarizing the paper**. Here, your focus should be on outlining the main contributions, highlighting key related works, providing an overview of the methodology, and delving into the key experiments conducted. For this part, you may find it beneficial to access the LaTeX source of the paper on arXiv (as "Other formats"). When incorporating tables and figures from the paper, prioritize the most pertinent ones, trimming any unnecessary details.

In the second section, your task is to **select another research paper** that bears significant relevance to your first paper. This could be a paper featuring a *competing* method, one that laid the *groundwork* for your paper, or even a subsequent *extension* of your paper. You must
    1. explain your choice,
    2. quickly summarize this second paper,
    3. discuss the relation with your first paper.

Academic dishonesty and intellectual laziness, such as relying on LLMs to completely generate your report or the speech of your presentation, will result in penalties. Please note that using LLMs to do light editing of your report is not forbidden.


## Additional info
You have 9-10 minutes (both students) to present your paper, and there will be 5 minutes of questions.
- **If you go beyond 10 minutes, the jury will stop you.** This is a sign of little preparation, so it will have an impact on the grade.
- The jury can choose which student should answer a specific question, so both students should understand the paper thoroughly.
- **For the questions, please go straight to the point**; don't go around with unnecessary details. If you don't understand the question, it's OK to ask for clarification. If you start answering something else, the jury will stop you. One good answer shouldn't take more than one minute.
- In the first email, I sent you the items that will be considered for the final grade. Please be sure to **provide the key technical contributions of the paper, explain why the paper is relevant for the literature (novelty), and provide and explain the key results that support the contributions**. The other three items are the **clarity** of the presentation, the **answers** to the questions, and the discussion of the **new experiment** you performed.
- The grade is individual.

### Document submission
- Please remember to **upload your presentations** in PDF by Monday, March 24 at most here https://partage.imt.fr/index.php/s/CaiWPT6ZkPTSEsr. Check here https://partage.imt.fr/index.php/s/7y5kkextZxpr4Mn.
    - name yourHour_yourDate_firstStudent_secondStudent.pdf, for example, 08h30_March25_MarieCurie_BlaisePascal.pdf. 
    - 13h45_March25_EdwinRoussin_TristanWaddington.pdf 
- Please upload your **final report** here: https://partage.imt.fr/index.php/s/gymYkMLBtpc7rae. The file should be named firstStudent_secondStudent.pdf. There's one report for both students, so you should select the same (second) paper for both students. Please include a small discussion about the new experiment you performed. You have until April 1st to submit your report.

# Personal notes
In graph theory, a **clique** is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent. In other words, a clique is a complete subgraph, meaning that there is an edge connecting every pair of vertices within the clique.

For example, in a graph network:
A 3-clique (or triangle) consists of three vertices, each connected to the other two.
A 4-clique consists of four vertices, each connected to the other three.
