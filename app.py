import streamlit as st
import pandas as pd
import joblib

# ---------- Classe de prédiction ----------
class FuelPredictor:
    _CATEGORIES = {
        "Company": [
            "Audi", "BMW", "Ferrari", "Ford", "Honda", "Hyundai", "Jaguar",
            "Kia", "Lamborghini", "Land Rover", "Mahindra", "Maruti", "Porsche",
            "Renault", "Rolls Royce", "Tata", "Toyota"
        ],
        "Type": ["Convertible", "Coupe", "Hatchback", "MUV", "SUV", "Sedan"],
        "Transmission": ["Automatic", "Manual", "Semi-Auto"]
    }

    _FEATURE_ORDER = [
        "Company", "Type", "Transmission", 
        "Engine", "Mileage", "Kms_driven", 
        "Horsepower (kw)", "Year", "Price (Lakhs)"
    ]

    def __init__(
        self,
        model_path="best_model.pkl",
        encoder_path="ordinal_encoder.pkl"
    ):
        self.pipeline = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

    def _prepare_dataframe(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                "Les données doivent être un dictionnaire ou un DataFrame"
            )
        df = df[self._FEATURE_ORDER]
        cat_cols = [
            "Company",
            "Type",
            "Transmission"
        ]
        df[cat_cols] = self.encoder.transform(df[cat_cols])
        return df

    def predict(self, data):
        X = self._prepare_dataframe(data)
        pred = self.pipeline.predict(X)
        return pred[0] if len(pred) == 1 else pred.tolist()

    def predict_proba(self, data):
        X = self._prepare_dataframe(data)
        probas = self.pipeline.predict_proba(X)
        classes = self.pipeline.classes_
        if len(probas) == 1:
            return {cls: prob for cls, prob in zip(classes, probas[0])}
        return [{cls: prob for cls, prob in zip(classes, row)} for row in probas]


# ---------- Fonction d'échange croisé (Swap) ----------
def swap_diesel_electrifie(value):
    """
    Échange exactement :
    - 'Electrifie' (ou variante) -> 'Diesel'
    - 'Diesel' -> 'Electrifie'
    """
    targets_electrifie = ["Electrifie", "Électrifié", "Electrique", "Électrique"]

    if isinstance(value, str):
        if value in targets_electrifie:
            return "Diesel"
        if value == "Diesel":
            return "Electrifie"
        return value

    if isinstance(value, dict):
        new_dict = {}
        for k, v in value.items():
            if k in targets_electrifie:
                new_key = "Diesel"
            elif k == "Diesel":
                new_key = "Electrifie"
            else:
                new_key = k
            new_dict[new_key] = v
        return new_dict

    return value


# ---------- Configuration globale & Interface Streamlit ----------
st.set_page_config(
    page_title="Estimation Carburant Véhicule", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    div[data-testid="stForm"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 24px;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return FuelPredictor("best_model.pkl")

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Impossible de charger le pipeline d'estimation : {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Estimation du type de carburant")
st.caption("Sélectionnez les caractéristiques techniques du véhicule pour obtenir l'estimation correspondante.")

tab_calculator, tab_history = st.tabs(["Calculateur", "Historique des données"])

with tab_calculator:
    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.subheader("Caractéristiques du véhicule")
        
        with st.form("prediction_form"):
            st.markdown("##### Informations générales")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                company = st.selectbox("Marque", predictor._CATEGORIES["Company"])
            with col_b:
                type_ = st.selectbox("Carrosserie", predictor._CATEGORIES["Type"])
            with col_c:
                transmission = st.selectbox("Transmission", predictor._CATEGORIES["Transmission"])

            st.markdown("##### Spécifications techniques")
            col_d, col_e = st.columns(2)
            with col_d:
                engine = st.number_input("Cylindrée (cm³)", min_value=0, value=1500, step=100)
                horsepower = st.number_input("Puissance (kW)", min_value=0, value=85, step=5)
                mileage = st.number_input("Consommation (km/L)", min_value=0.0, value=18.0, step=0.1)
            with col_e:
                kms_driven = st.number_input("Kilométrage (km)", min_value=0, value=45000, step=1000)
                year = st.slider("Année de mise en circulation", min_value=2000, max_value=2026, value=2018)
                price = st.number_input("Prix estimé (Lakhs)", min_value=0.0, value=8.5, step=0.5)

            submitted = st.form_submit_button("Calculer l'estimation", use_container_width=True)

    with col_result:
        st.subheader("Résultats de l'analyse")

        if submitted:
            car_data = {
                "Company": company,
                "Type": type_,
                "Transmission": transmission,
                "Engine": engine,
                "Mileage": mileage,
                "Kms_driven": kms_driven,
                "Horsepower (kw)": horsepower,
                "Year": year,
                "Price (Lakhs)": price
            }

            with st.spinner("Analyse en cours..."):
                raw_prediction = predictor.predict(car_data)
                raw_probas = predictor.predict_proba(car_data)

                # Échange direct des deux catégories
                prediction = swap_diesel_electrifie(raw_prediction)
                probas = swap_diesel_electrifie(raw_probas)

            max_proba = probas[prediction]
            
            st.metric(
                label="Type de carburant estimé", 
                value=str(prediction).upper(),
                delta=f"Indice de confiance : {max_proba:.1%}"
            )
            
            st.divider()
            st.markdown("##### Répartition de la probabilité")
            
            for fuel, prob in sorted(probas.items(), key=lambda x: x[1], reverse=True):
                st.text(f"{fuel} — {prob:.1%}")
                st.progress(float(prob))

            st.session_state.history.insert(0, {
                "Company": company,
                "Type": type_,
                "Transmission": transmission,
                "Year": year,
                "Engine": engine,
                "Prediction": prediction,
                "Confiance": f"{max_proba:.1%}"
            })
            if len(st.session_state.history) > 5:
                st.session_state.history.pop()

        else:
            st.info("Veuillez remplir les informations à gauche puis cliquer sur 'Calculer l'estimation' pour afficher le résultat.")

with tab_history:
    st.subheader("Dernières demandes traitées")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if st.button("Réinitialiser l'historique"):
            st.session_state.history = []
            st.rerun()
    else:
        st.text("Aucun historique disponible pour le moment.")
