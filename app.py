import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'model.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / 'test.csv')
    with open(BASE_DIR / 'config.json', 'r') as f:
        config = json.load(f)
    return df, config

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
df_test, config = load_data()

X_test = df_test.drop(columns=['target'])
y_test = df_test['target']
features = X_test.columns.tolist()

y_proba = test_proba(model, X_test)

st.title("Сервис рекомендаций вин")
st.sidebar.header("Настройки")
default_threshold = config.get('threshold', 0.5)
threshold = st.sidebar.slider("Порог вероятности", 0.0, 1.0, float(default_threshold), 0.01)
st.session_state['threshold'] = threshold

tab1, tab2, tab3, tab4 = st.tabs(["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])

with tab1:
    st.subheader("Введите характеристики вина")
    col1, col2 = st.columns(2)
    input_data = {}

    for i, feature in enumerate(features):
        f_min = float(X_test[feature].min())
        f_max = float(X_test[feature].max())
        f_median = float(X_test[feature].median())
        
        with col1 if i % 2 == 0 else col2:
            if feature == 'is_red':
                input_data[feature] = float(st.radio(f"{feature} (1 - красное, 0 - белое)", options=[0, 1], horizontal=True))
            else:
                input_data[feature] = st.slider(feature, f_min, f_max, f_median, (f_max - f_min) / 100 if f_max != f_min else 1.0)

    if st.button("Оценить вино", type="primary"):
        input_df = pd.DataFrame([input_data])[features]
        proba = model.predict_proba(input_df)[0, 1]
        decision = "Хорошее (Рекомендовать)" if proba >= threshold else "Плохое (Не рекомендовать)"
        color = "green" if proba >= threshold else "red"
        
        st.markdown("---")
        st.metric(label="Вероятность, что вино хорошее", value=f"{proba:.2%}")
        st.markdown(f"### Решение модели: :{color}[{decision}]")

with tab2:
    st.subheader(f"Анализ модели при пороге {threshold:.2f}")
    
    y_pred = (y_proba >= threshold).astype(int)
    
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Precision", f"{p:.3f}")
    col_met2.metric("Recall", f"{r:.3f}")
    col_met3.metric("F1-score", f"{f1:.3f}")
    
    st.markdown("---")
    
    col_cm, col_text = st.columns([1, 1])
    with col_cm:
        st.write("**Матрица ошибок**")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Плохое', 'Хорошее'])
        fig_cm, ax_cm = plt.subplots()
        disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
        st.pyplot(fig_cm)
        
    with col_text:
        st.write("**Рекомендация для сомелье**")
        if threshold > 0.5:
            st.info("Порог высокий. Сервис рекомендует только те вина, в которых модель очень уверена. Подборка станет **узкой**, но в ней почти не будет плохих вин.")
        elif threshold < 0.3:
            st.warning("Порог низкий. Сервис будет рекомендовать почти всё подряд. Подборка станет **очень широкой**, но в ней будет много посредственных вин.")
        else:
            st.success("Сбалансированный порог. Сервис находит достаточно хороших вин, не слишком засоряя подборку плохими.")

    st.markdown("---")
    
    col_roc, col_pr = st.columns(2)
    
    with col_roc:
        st.write("**ROC-кривая**")
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_proba):.3f}")
        idx_roc = np.argmin(np.abs(thresholds_roc - threshold))
        ax_roc.scatter(fpr[idx_roc], tpr[idx_roc], color='red', s=100, zorder=5, label=f'Порог {threshold:.2f}')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend()
        st.pyplot(fig_roc)
        
    with col_pr:
        st.write("**PR-кривая**")
        prec, rec, thresholds_pr = precision_recall_curve(y_test, y_proba)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(rec, prec, label="PR Curve")
        idx_pr = np.argmin(np.abs(thresholds_pr - threshold))
        ax_pr.scatter(rec[idx_pr], prec[idx_pr], color='red', s=100, zorder=5, label=f'Порог {threshold:.2f}')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        st.pyplot(fig_pr)

with tab3:
    st.subheader("Загрузка файла с винами")
    st.write("Загрузите CSV-файл с признаками вин. В файле **не должно быть** столбца `target`.")
    
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload.columns = df_upload.columns.str.strip().str.lower()
            
            missing_cols = [col for col in features if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"Ошибка: в загруженном файле не хватает следующих столбцов: {', '.join(missing_cols)}")
            else:
                X_upload = df_upload[features]
                probas = model.predict_proba(X_upload)[:, 1]
                decisions = (probas >= threshold).astype(int)
                
                result_df = df_upload.copy()
                result_df['probability_good'] = np.round(probas, 4)
                result_df['decision'] = np.where(decisions == 1, 'Хорошее', 'Плохое')
                
                st.write(f"**Результаты предсказания (порог {threshold:.2f}):**")
                st.dataframe(result_df.head(20))
                
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты (CSV)",
                    data=csv_data,
                    file_name='wine_predictions.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")

with tab4:
    st.subheader("Коэффициенты логистической регрессии")
    st.write("Диаграмма показывает, как каждый признак влияет на вероятность того, что вино будет признано хорошим.")
    
    try:
        clf = model.named_steps['logisticregression']
        coefs = clf.coef_[0]
        
        coef_df = pd.DataFrame({'Признак': features, 'Коэффициент': coefs})
        coef_df = coef_df.sort_values(by='Коэффициент', ascending=True)
        
        fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
        colors = ['red' if c < 0 else 'green' for c in coef_df['Коэффициент']]
        ax_coef.barh(coef_df['Признак'], coef_df['Коэффициент'], color=colors)
        ax_coef.set_xlabel("Значение коэффициента")
        ax_coef.axvline(0, color='black', linewidth=0.8)
        st.pyplot(fig_coef)
        
        pos_features = coef_df[coef_df['Коэффициент'] > 0].sort_values(by='Коэффициент', ascending=False)['Признак'].tolist()
        neg_features = coef_df[coef_df['Коэффициент'] < 0].sort_values(by='Коэффициент', ascending=True)['Признак'].tolist()
        
        st.markdown("---")
        st.write("**Интерпретация:**")
        if pos_features:
            st.write(f"Сильнее всего за то, что вино хорошее, голосуют признаки: **{', '.join(pos_features[:3])}**.")
        if neg_features:
            st.write(f"Сильнее всего против того, что вино хорошее, голосуют признаки: **{', '.join(neg_features[:3])}**.")
            
    except Exception as e:
        st.error(f"Не удалось получить коэффициенты модели: {e}")

# ОТВЕТ НА ВОПРОС 4.5
# """
# Ответ на вопрос 4.5:
# 1. Для модели используется st.cache_resource, потому что объект модели (особенно с внутренними весами и структурой) не является простыми данными (как таблица). cache_resource создает один экземпляр объекта на весь сервер и переиспользует его, что экономит память и время на десериализацию (joblib.load). 
# Для таблиц используется st.cache_data, так как это просто данные (DataFrame), которые Streamlit умеет эффективно хешировать и хранить в виде копий, чтобы избежать их случайного изменения.
# 2. Если вызвать joblib.load без декоратора, то при каждом действии пользователя (например, при сдвиге ползунка порога) скрипт app.py будет перезапускаться сверху вниз. Это приведет к тому, что модель будет заново читаться с жесткого диска на каждом шаге. Это вызовет жуткие тормоза.
# 3. Вероятности считаются один раз благодаря функции test_proba с декоратором st.cache_data. Модель передается с подчеркиванием (_model), чтобы Streamlit не пытался ее хешировать (так как она не хешируемая), а просто использовал для расчета. При движении ползунка меняется только сравнение y_proba >= threshold, а сам predict_proba больше не вызывается.
# """