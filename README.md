# arize_assessment
1. Build an LLM application. You can use Langchain, Llamindex, OpenAI, etc. You create this application in a notebook or hosted environment.
2. Tasked to evaluate Arize's Phoenix: ML Observability Platform for Developers:
  a. How was Onboarding Experience? What went well, what didn't?
  b. What metrics would you measure for new users?
3. Come back with ideas and the rationale on what you would invest in if you were the PM of this team.
________________________________________________________________________________________________________________________________________________

2. a.
- the latter command in !pip install arize-phoenix arize["AutoEmbeddings"] should be pip install 'arize[AutoEmbeddings]’ in the docs
- Required a lot of older dependencies (e.g. grpcio==1.34.0, numpy==1.19.2, six==1.15.0), which isn't a huge problem, but if not running on Virtual Environment was kind of annoying
- generate_embeddings() runtime is very slow compared to putting it into OpenAI API call. Not sure if there is much we can change there, but took 20-30 seconds to run each time on dataset with relatively small amount of tokens (<1000).
- In terms of docs and ease of use as an intermediate coder, found Langsmith's docs much more comprehensive. However, in the end, I believe Arize's examles were relatively good. Langsmith's minimalist UI drew my attention. However, if target market is already developers, should not matter much and should be put on the backburner of considerations. However,once onboarded, I did enjoy the Arize UI/platform more, as it seemed more cohesive/comprehensive. Especially when learning about both features, I found it much easier to migrate onto Arize. Just need to get the eyes there.
  
   b.
- Support for model explainability metrics: 
- Integration with more machine learning frameworks and libraries:
- Automated model retraining capabilities:
- Ability to track and visualize model lineage and versioning: Already has this as I have seen on both Langsmith and Arize, but feel like Arize can do better in terms of this.

3. automated suggestions for feature engineering based on data patterns, and real-time monitoring and alerting capabilities for immediate issue resolution.

