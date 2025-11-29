import streamlit as st
import joblib
import numpy as np
import re
import string
import pandas as pd
import altair as alt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ==============================
# 0. Метрики моделей (из ноутбука)
# ==============================

MODEL_METRICS = {
    "Logistic Regression": {
        "accuracy": 0.794,
        "precision": 0.794,
        "recall": 0.794,
        "f1": 0.794,
    },
    "Linear SVM (calibrated)": {
        "accuracy": 0.790,
        "precision": 0.791,
        "recall": 0.790,
        "f1": 0.790,
    },
    "Multinomial Naive Bayes": {
        "accuracy": 0.767,
        "precision": 0.767,
        "recall": 0.767,
        "f1": 0.767,
    },
}
# Короткие имена моделей — используем только для графиков
SHORT_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Linear SVM (calibrated)": "Linear SVM",
    "Multinomial Naive Bayes": "Naive Bayes",
}



# ==============================
# 1. Загрузка моделей
# ==============================

@st.cache_resource
def load_models():
    """
    Загружаем обученный TF-IDF и три модели:
    - Logistic Regression
    - Linear SVM (калиброванный)
    - Multinomial Naive Bayes

    Добавлена обработка ошибок: если файла нет — показать
    сообщение в интерфейсе, а не молча падать.
    """
    try:
        tfidf = joblib.load("artifacts/tfidf_vectorizer.joblib")

        models = {
            "Logistic Regression": joblib.load("artifacts/logreg_best.joblib"),
            "Linear SVM (calibrated)": joblib.load("artifacts/linear_svc_best.joblib"),
            "Multinomial Naive Bayes": joblib.load("artifacts/mnb_best.joblib"),
        }

    except FileNotFoundError as e:
        st.error(
            "❌ **Не найдены файлы моделей в папке `artifacts/`.**\n"
            "Убедитесь, что вы **обучили и сохранили артефакты** (joblib-файлы)."
        )
        raise e

    return tfidf, models


# ==============================
# 2. Инструменты препроцессинга
# ==============================

@st.cache_resource
def get_text_tools():
    """
    Загружаем всё нужное для препроцессинга:
    стоп-слова, стеммер и токенизатор (как в ноутбуке).
    """
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # Тот же токенизатор, что и в ноутбуке
    re_tok = re.compile(f'([{string.punctuation}“”¨ «»®´·º½¾¿¡§£₤‘’])')

    def tokenize(s: str):
        return re_tok.sub(r" \1 ", s).split()

    return stop_words, ps, tokenize


# ==============================
# 3. Препроцессинг и предсказание
# ==============================

def preprocess_text(text: str, stop_words, ps, tokenize) -> str:
    """
    Полный препроцессинг:
    - токенизация
    - удаление стоп-слов
    - стемминг
    Возвращает очищенную строку.
    """
    tokens = [w for w in tokenize(text) if w.lower() not in stop_words]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)


def predict_sentiment(text: str, tfidf, model, stop_words, ps, tokenize):
    """
    Делает предсказание тональности для одного текста.

    Возвращает (prob_pos, label, text_clean), где:
    - prob_pos    — вероятность положительного класса
    - label       — 1 (positive) или 0 (negative)
    - text_clean  — текст после препроцессинга
    """
    text_clean = preprocess_text(text, stop_words, ps, tokenize)
    X_vec = tfidf.transform([text_clean])

    prob_pos = model.predict_proba(X_vec)[0, 1]
    label = int(prob_pos >= 0.5)

    return prob_pos, label, text_clean


# ==============================
# 4. Интерфейс Streamlit
# ==============================

def main():
    st.set_page_config(
        page_title="Классификация текста по настроению",
        page_icon="💬",
        layout="wide",
    )

    # ----- Заголовок и описание -----
    st.title("💬 Классификация текста по настроению")

    st.markdown(
        """
        Это приложение по анализу тональности текстов (отзывы, твиты, комментарии) на английском языке.

        Модели обучены на больших корпусах отзывов с использованием TF-IDF-признаков.
        Ниже вы можете выбрать модель и проверить, как она оценивает ваш текст.
        """
    )
    st.divider()
    # Инициализируем историю запросов
    if "history" not in st.session_state:
        st.session_state["history"] = []


    # Загружаем модели и инструменты препроцессинга
    tfidf, models = load_models()
    stop_words, ps, tokenize = get_text_tools()

    # ================= SIDEBAR: выбор модели и метрики =================
    st.sidebar.title("⚙️ Настройки")

    model_name = st.sidebar.radio(
        "Выберите модель",
        list(models.keys()),
        index=0,
    )
    current_model = models[model_name]

    metrics = MODEL_METRICS.get(model_name)
    if metrics:
        st.sidebar.markdown("### 📊 Качество модели")
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.sidebar.metric("F1 (weighted)", f"{metrics['f1']:.3f}")
        st.sidebar.caption("Метрики взяты из оффлайн-обучения моделей.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Модели обучены на очищенных англоязычных отзывах.")

    # ================= Вкладки =================
    tab_single, tab_compare, tab_train = st.tabs(
        ["🔍 Анализ одного текста", "📊 Сравнение моделей", "📚 Обучение моделей"]
    )

    # ---------- Вкладка 1: анализ одного текста ----------
    with tab_single:
        st.subheader("1. Введите текст отзыва (на английском)")

        # состояние для примеров/очистки
        if "input_text" not in st.session_state:
            st.session_state["input_text"] = ""

        # кнопки с примерами
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            if st.button("👍 Позитивный пример"):
                st.session_state["input_text"] = (
                    "This movie was really good, I enjoyed every minute of it!"
                )
        with col_ex2:
            if st.button("👎 Негативный пример"):
                st.session_state["input_text"] = (
                    "This movie was terrible and boring, I would not recommend it."
                )
        with col_ex3:
            if st.button("🧹 Очистить"):
                st.session_state["input_text"] = ""

        user_text = st.text_area(
            "Текст",
            height=180,
            placeholder="For example: This movie was surprisingly good...",
            value=st.session_state["input_text"],
        )

        # маленький индикатор длины текста
        st.caption(f"Длина текста: {len(user_text.split())} слов")

        st.subheader("2. Предсказание настроения")
        st.caption(f"Используется модель: **{model_name}**")

        if st.button("🔍 Определить настроение"):
            if not user_text.strip():
                st.warning("Сначала введите текст.")
            else:
                prob_pos, label, text_clean = predict_sentiment(
                    user_text, tfidf, current_model, stop_words, ps, tokenize
                )

                sentiment = "Положительный" if label == 1 else "Отрицательный"
                prob_neg = 1.0 - prob_pos

                # --- карточка результата ---
                col_res_left, col_res_right = st.columns([2, 1])
                with col_res_left:
                    st.success(f"Результат: **{sentiment}**")
                    st.write(
                        f"Вероятность положительного класса: "
                        f"`{prob_pos:.3f}`, отрицательного: `{prob_neg:.3f}`"
                    )
                with col_res_right:
                    st.metric("p(positive)", f"{prob_pos:.3f}")
                    st.metric("p(negative)", f"{prob_neg:.3f}")

                # --- комментарий по уверенности ---
                if prob_pos > 0.9 or prob_pos < 0.1:
                    st.info("Модель довольно уверена в своём ответе.")
                else:
                    st.info(
                        "Модель не полностью уверена — текст может быть нейтральным "
                        "или содержать смешанные формулировки."
                    )

                # --- показываем, как текст изменился после препроцессинга ---
                with st.expander("🔧 Пошаговая обработка текста"):
                    st.markdown("**Исходный текст:**")
                    st.write(user_text)

                    st.markdown("**После предобработки (стоп-слова, стемминг):**")
                    st.write(text_clean)

                # ---------- сохраняем в историю ----------
                short_text = user_text.strip().replace("\n", " ")
                if len(short_text) > 80:
                    short_text = short_text[:77] + "..."

                st.session_state["history"].insert(
                    0,
                    {
                        "text": user_text,
                        "preprocessed": text_clean,
                        "short_text": short_text,
                        "model": model_name,
                        "sentiment": sentiment,
                        "prob_pos": prob_pos,
                        "prob_neg": prob_neg,
                    },
                )
                # храним только последние 5 записей
                st.session_state["history"] = st.session_state["history"][:5]

                # ---------- выводим историю + кнопка очистки ----------
                if st.session_state["history"]:
                    st.markdown("#### История последних запросов")

                    # кнопка очистки истории
                    if st.button("🧹 Очистить историю", key="clear_history"):
                        st.session_state["history"] = []
                    else:
                        for i, item in enumerate(st.session_state["history"], start=1):
                            with st.expander(
                                f"{i}. {item['sentiment']} "
                                f"(*{item['model']}*, p_pos={item['prob_pos']:.3f}) — "
                                f"«{item['short_text']}»"
                            ):
                                st.markdown("**Исходный текст:**")
                                st.write(item["text"])

                                st.markdown("**После предобработки:**")
                                st.write(item["preprocessed"])



    # ---------- Вкладка 2: сравнение моделей ----------
    with tab_compare:
        st.subheader("Сравнение моделей по метрикам")

        df_metrics = pd.DataFrame(MODEL_METRICS).T  # index = название модели
        df_metrics = df_metrics[["accuracy", "precision", "recall", "f1"]].round(3)
        st.dataframe(df_metrics, use_container_width=True)

        st.markdown("#### F1-score для всех моделей")

        # Строим датафрейм для графика и добавляем короткие имена
        df_plot = df_metrics.reset_index().rename(columns={"index": "model"})
        df_plot["model_short"] = df_plot["model"].map(SHORT_NAMES)

        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                # по оси X используем короткое имя, в tooltip — полное
                x=alt.X(
                    "model_short:N",
                    sort=None,
                    axis=alt.Axis(title="Модель", labelAngle=0),
                ),
                y=alt.Y("f1:Q", title="F1-score"),
                tooltip=["model", "f1"],
            )
            .properties(height=280)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Кратко о моделях")

        st.markdown(
            """
            - **Logistic Regression** — линейная модель, хорошо работает на TF-IDF-признаках,
              даёт интерпретируемые веса для слов.
            - **Linear SVM (calibrated)** — опорные векторы с последующей калибровкой
              вероятностей (CalibratedClassifierCV, метод *sigmoid*).
            - **Multinomial Naive Bayes** — простая и быстрая модель для текстов,
              часто используется как сильная базовая линия.
            """
        )

        with st.expander("Что такое TF-IDF?"):
            st.write(
                """
                TF-IDF (Term Frequency – Inverse Document Frequency) —
                способ оценить «важность» слова в документе относительно всего корпуса.
                Часто встречающиеся во всех документах слова (типа *the, and, of*)
                получают малый вес, а слова, характерные именно для данного документа —
                более высокий.
                """
            )

        with st.expander("Что делает препроцессинг текста в этой работе?"):
            st.markdown(
                """
                1. Разбиваем текст на токены (слова и знаки пунктуации).  
                2. Удаляем английские стоп-слова (*the, and, of, ...*).  
                3. Применяем стемминг (PorterStemmer), приводя слова к основе  
                   (*liked → like, movies → movi*).  
                4. Преобразуем текст в TF-IDF вектор и подаём в выбранную модель.
                """
            )

    # ---------- Вкладка 3: Обучение моделей ----------
    with tab_train:
        st.subheader("Общая схема обучения моделей")

        st.markdown(
            """
            **1. Данные**

            - Используются датасеты отзывов на английском языке с метками настроения (0 — негатив, 1 — позитив).
            - Данные предварительно очищены: убраны HTML-теги, ссылки, спецсимволы, дубликаты и т.д.

            **2. Предобработка текста**

            - приведение к нижнему регистру  
            - токенизация с помощью регулярных выражений  
            - удаление английских стоп-слов (NLTK `stopwords`)  
            - стемминг (PorterStemmer)  
            - обратное склеивание токенов в строку  

            **3. Векторизация (TF-IDF)**

            - Используется `TfidfVectorizer` из `scikit-learn`.  
            - Ограничение по минимальной/максимальной частоте термов, `ngram_range=(1, 5)`.  
            - На выходе — разреженная матрица признаков для моделей.

            **4. Модели**

            - **Logistic Regression** (`LogisticRegression`)  
              - подбор гиперпараметров по сетке (C, penalty, solver, class_weight) через `GridSearchCV`.  
            - **Linear SVM (calibrated)**  
              - базовая модель `LinearSVC`,  
              - обёртка `CalibratedClassifierCV(method="sigmoid")` для получения вероятностей,  
              - возможный поиск по параметру `C`.  
            - **Multinomial Naive Bayes** (`MultinomialNB`)  
              - альфа-параметр сглаживания подобран экспериментально.

            **5. Оценка качества**

            - Разделение данных на train/test.  
            - Метрики: *accuracy, precision, recall, F1* (взвешенные).  
            - Дополнительно строились матрицы ошибок и ROC-кривые (в ноутбуке).

            **6. Сохранение артефактов**

            - Обученный TF-IDF-векторизатор и лучшие модели сохраняются с помощью `joblib.dump(...)` в папку `artifacts/`.  
            - В этом приложении они загружаются при старте и используются для онлайн-предсказаний.
            """
        )



if __name__ == "__main__":
    main()

