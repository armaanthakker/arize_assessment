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
- Support for model explainability metrics: Arize does a great job of including multiple explainability metrics such as Feature Importance, SHAP vals, and LIME. However when using Langsmith, I noticed their explainability metrics are much more compact and comprehensive. First of all, their metrics are much more user facing such as fidelity score, consistency, robustness, human satisfaction, etc. Arize focuses much more on data drift and modeling predictions instead of the explanations itself. I feel integrating a larger scale of explainability metrics could be helpful and give Arize a competitive advantage. Addtiionally, each instance in Langsmith had a platform to see all the metrics together while you may not be able to view all of the metrics together in Arize's platform.
- Integration with more machine learning frameworks and libraries: Currently, Phoenix has integration capabilities with popular frameworks and libraries such as TensorFlow, PyTorch, Scikit-learn, and XGBoost. However, it does not have integration capabilities with frameworks and libraries like Keras, H2O.ai, and LightGBM.
- Automated model retraining capabilities: I feel this would be a cool feature to have. With drift detection and model performance metrics, I feel like Phoenix already has the tools necesary to implement an automatic retraining model. It would also streamline the model maintenence process for users making Phoenix much more attractive.

3.
- automated suggestions for feature engineering based on data patterns
- real-time monitoring and alerting capabilities for immediate issue resolution.

