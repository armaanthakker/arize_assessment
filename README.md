# arize_assessment
1. Build an LLM application. You can use Langchain, Llamindex, OpenAI, etc. You create this application in a notebook or hosted environment.
2. Tasked to evaluate Arize's Phoenix: ML Observability Platform for Developers:
  a. How was Onboarding Experience? What went well, what didn't?
  b. What metrics would you measure for new users?
3. Come back with ideas and the rationale on what you would invest in if you were the PM of this team.
________________________________________________________________________________________________________________________________________________

Since our goal is to evaluate developers experience with Phoenix, I will first focus on tactical onboarding feedback to provide a smooth install and quick time to value, and contrast it with Langsmith. The working assumption is that we need to have developers see the “aha moment” quickly in their onboarding cycle so they can then collaborate or hand-off to ML Engineers where Arize truly shines over any competitor. The more developers are bought into Phoenix, the more they adopt it as LLM apps proliferate and provide a pathway to ML Engineers to adopt the full feature Arize platform. My second section will focus on strategic product feedback to illustrate our value to developers and ML Engineers and some areas to improve our competitive differentiation over Langsmith further.


2. a.
- the latter command in !pip install arize-phoenix arize["AutoEmbeddings"] should be pip install 'arize[AutoEmbeddings]’ in the docs
- Required a lot of older dependencies (e.g. grpcio==1.34.0, numpy==1.19.2, six==1.15.0), which isn't a huge problem, but if not running on Virtual Environment was kind of annoying
- generate_embeddings() runtime is very slow compared to putting it into OpenAI API call. Not sure if there is much we can change there, but took 20-30 seconds to run each time on dataset with relatively small amount of tokens (<1000).
- In terms of docs and ease of use as an intermediate coder, found Langsmith's docs much more comprehensive. However, in the end, I believe Arize's examples were relatively good. Langsmith's minimalist UI drew my attention. However, if target market is already developers, should not matter much and should be put on the backburner of considerations. However,once onboarded, I did enjoy the Arize UI/platform more, as it seemed more cohesive/comprehensive. Especially when learning about both features, I found it much easier to migrate onto Arize. Just need to get the eyes there.
  
   b.
- Support for model explainability metrics: Arize does a great job of including multiple explainability metrics such as Feature Importance, SHAP vals, and LIME. However when using Langsmith, I noticed their explainability metrics are much more compact and comprehensive. First of all, their metrics are much more user facing such as fidelity score, consistency, robustness, human satisfaction, etc. Arize focuses much more on data drift and modeling predictions instead of the user-facing explanations/resposnes themselves. I feel integrating a larger scale of explainability metrics could be helpful and give Arize a competitive advantage. Specific metrics I would include would be  Simplicity(evaluating how complex the response is), Explanation Confidence(Quantifying the confidence or certainty of model explanations, indicating how confident the model is in its predictions and explanations. Higher confidence levels increase trust in the model's decisions), Explanation Certainty Under Data Shifts (how well model explanations generalize to new or unseen data distributions. Models with explanations that remain consistent under data shifts are more reliable in real-world scenarios). Other than that, another cool idea could be Subset Feature Stability (how consistent is a feature or cluster of features accross different subsets of the same dataset. More stable features could be better-saw this online).----Did not realize that you could define your own metrics. Maybe integrate these into Arize Metrics.

- Arize does not have a feature for integrating with AutoML platforms. In order to make the platform more sticky and more reachable to a wide array of MLEngineers, I feel that Arize can do a better job integrating with these popular AutoML platforms such as AWS Sagemaker or Kubeflow. By integrating with these platformns 

- Arize can integrate with more MSP
- Addtionally, each instance in Langsmith had a platform to see all the metrics together while you may not be able to view all of the metrics together in Arize's platform. Maybe create a UI where user can view all of the metrics in one compact space.

3.
- Automated suggestions for feature engineering based on data patterns: Currently, Phoenix points out feature based and cluster based anomolies in data, helping a lot with trouble-shooting. However, in terms of being proactive, Phoenix could build an algorithm to analyze the characteristics of the input data and recommend potential features that could improve model performance. While this may be prevalant or even automated for some features such as dimension reduction and cluster analysis, the ability for Phoenix to automate the feature engineering process fully could be very helpful and make the product a much more vertical fit.
- Data Quality Monitoring: Currently, Arize does not monitor the quality of the inputted data of the model. I ran multiple models with poor self-made data and Phoenix detected a lot of drift everywhere. Instead, by preemptively checking for inputted data quality, it could save users a lot more time and make the product more vertical/sticky as well.
- Automated model retraining capabilities: I feel this would be a cool feature to have. With drift detection and model performance metrics, I feel like Phoenix already has the tools necesary to implement an automatic retraining model. It would also streamline the model maintenence process for users making Phoenix much more attractive. ----Maybe feed the outlier/drift data back into the model automatically. So user does not have to do this. (Not sure if automatic retraining traces back to the actual developer however. If they link their data back to a continuously updating source, this point might not make a lot of sense).
- Integration with more machine learning frameworks and libraries: Currently, Phoenix has integration capabilities with popular frameworks and libraries such as TensorFlow, PyTorch, Scikit-learn, and XGBoost. However, it does not have integration capabilities with frameworks and libraries like Keras, H2O.ai, and LightGBM.
