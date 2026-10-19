# Movie Success Predictor

A Streamlit app that estimates how a movie will perform before it is released. Enter a production budget and the expected ratings, and the app predicts the movie's revenue, profit, ROI and its chance of success.

> A second version with a bolder design lives in [movie-dashboard-neon-edition](https://github.com/Rawhy7/movie-dashboard-neon-edition).

## How it works

- Two **Random Forest** models from scikit-learn are trained on a dataset of movies (`Dataset Authentic Movies.xlsx`):
  - a **regressor** that predicts global box office revenue from the budget
  - a **classifier** that predicts the chance of success from the budget and the ratings
- Inputs in the sidebar:
  - Production budget ($1M – $350M)
  - Expected IMDb rating (1 – 10)
  - Expected Rotten Tomatoes score (0 – 100%)
- Results:
  - A verdict (from *High Risk* to *Strong Contender*) with the success probability
  - Predicted revenue, profit and ROI
  - Reception score, break-even point and the gap between critics and audience
  - Short, rule-based insights on the financial outlook and reception

## Built with

Python · Pandas · NumPy · scikit-learn · Streamlit

## Run it locally

```bash
git clone https://github.com/Rawhy7/movie-dashboard.git
cd movie-dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

## Author

**Abdelrahman Rawhy** – [Website](https://abdelrahmanrawhy.com) · [LinkedIn](https://www.linkedin.com/in/abdelrahman-rawhy/)
