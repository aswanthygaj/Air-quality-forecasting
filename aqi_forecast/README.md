# Air Quality Forecasting – Indian Cities
Run order (from this folder):
1. pip install -r requirements.txt   (python 3.10+)
2. DATA (pick one)
   a) quick start:  python src/make_synthetic_data.py      (already done: data/ has synthetic files)
   b) real data:    download Kaggle "Air Quality Data in India (2015-2020)" -> put city_day.csv in data/
                    python src/fetch_weather.py             (Open-Meteo weather, no key)
3. python src/prepare.py
4. python src/train_lstm.py          # trains + prints MAE/RMSE vs persistence
5. python src/prophet_baseline.py Delhi
6. streamlit run app.py
Optional: export OPENAQ_API_KEY=...  (live tab);  python src/train_tft.py (TFT).
