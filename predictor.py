import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 模型构建代码重构
# 1. 加载模型和数据
model = joblib.load('XGBoost.pkl')
X_test = pd.read_csv('X_test.csv')

# 2. 定义特征名称（与训练时保持一致）
feature_names = [
    "SLC6A13", "ANLN", "MARCO", "SYT13", "ARG2", "MEFV", "ZNF29P",
    "FLVCR2", "PTGFR", "CRISP2", "EME1", "IL22RA2", "SLC29A4",
    "CYBB", "LRRC25", "SCN8A", "LILRA6", "CTD_3080P12_3", "PECAM1"
]

# 3. 设置网页标题和说明
st.title("Non-small Cell Lung Cancer Risk Prediction Model")
st.markdown("Assessing the Risk of Non-Small Cell Lung Cancer Based on Diabetes-Related Gene Expression Levels.")

# 4. 创建输入表单（简化为滑块形式）
st.sidebar.header("Gene Expression Level Settings")
inputs = {}
for feature in feature_names:
    inputs[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=100000.0, value=161.0)

# 5. 显示输入数据表格
st.subheader("Input Gene Expression Data")
input_df = pd.DataFrame([inputs.values()], columns=inputs.keys())
st.table(input_df.style.highlight_max(axis=0))

# 6. 预测功能
if st.button("Calculate Disease Risk"):
    # 6.1 进行预测
    predicted_proba = model.predict_proba(input_df)[0]
    tumor_risk = predicted_proba[1] * 100  
    
    # 6.2 显示风险评估结果
    st.subheader("Risk Assessment Results")
    if tumor_risk >= 50:
        risk_level = "High Risk"
        color = "#FF5733"
    else:
        risk_level = "Low Risk"
        color = "#33C1FF"
    
    st.markdown(f"The probability of you having non-small cell lung cancer is...: <span style='color:{color}; font-weight:bold;'>{tumor_risk:.2f}%</span>", 
                unsafe_allow_html=True)
    st.markdown(f"Risk Level: <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", 
                unsafe_allow_html=True)
    
    # 6.3 医学建议
    if tumor_risk >= 50:
        advice = """
        We're sorry to inform you that, according to the model's prediction, you have a high risk of having the disease. It's advisable to contact a healthcare professional for a thorough examination at the earliest. Please note that our results are for reference only and cannot replace a professional diagnosis from a hospital.
        """
    else:
        advice = """
        We're glad to inform you that, according to the model's prediction, your disease risk is low. If you experience any discomfort, it's still advisable to consult a doctor. Please maintain a healthy lifestyle and have regular medical check-ups. Wishing you good health.
        """
    st.markdown(advice)
    
    # 7. SHAP解释功能
    st.subheader("SHAP")
    
    # 7.1 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # 7.2 处理多分类情况
    if isinstance(shap_values, list):
        shap_class_values = shap_values[1][0]  # 提取正类的SHAP值
        base_value = explainer.expected_value[1]  # 提取正类的基准值
    else:
        shap_class_values = shap_values[0]
        base_value = explainer.expected_value
    
    # 7.3 自定义SHAP图展示
    plt.figure(figsize=(12, 6))
    
    # 计算模型预测值与基准值的差异
    model_output = np.log(predicted_proba[1] / (1 - predicted_proba[1]))  # 将概率转换为log-odds
    difference = model_output - base_value
    
    # 创建自定义SHAP图
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.scatter([0] * len(feature_names), np.arange(len(feature_names)), color='blue', s=100)
    plt.yticks(np.arange(len(feature_names)), feature_names)
    
    # 绘制特征对模型输出的影响
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_class_values)):
        plt.plot([0, shap_val], [i, i], color='red' if shap_val > 0 else 'green', linewidth=3)
        plt.scatter([shap_val], [i], color='red' if shap_val > 0 else 'green', s=200)
    
    # 添加基准值和模型输出
    plt.scatter([base_value], [-1], color='black', s=200, marker='X', label='Base Value')
    plt.scatter([model_output], [-1], color='purple', s=200, marker='X', label='Model Output')
    plt.plot([base_value, model_output], [-1, -1], color='gray', linestyle='-', linewidth=1)
    
    plt.title('SHAP Values for Non-Small Cell Lung Cancer Risk Prediction', fontsize=14)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    
    # 7.4 显示SHAP图并清理
    st.pyplot(plt.gcf())
    plt.clf()