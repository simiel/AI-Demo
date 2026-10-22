# Customer Value Prediction

A local demo that predicts a customer's monetary value from frequency and RFM rank features and serves the result in a small web form. Samuel Mensah ([@simiel](https://github.com/simiel)) built it as one path from an e-commerce event analysis to a saved scikit-learn model: Jupyter notebooks engineer the customer table, `model.py` fits a random forest, and `app.py` scores four numbers and prints the prediction in cedis.

The page is a localhost interface for trying that saved model. The notebooks are the analysis behind it: brand mix, view timing, and the RFM table the forest is trained on.

## Features

- A Flask page titled **Customer Value Prediction** with a four-field form and a prediction readout.
- A **Get User** action that prefills the form with a random name and random integers so the form can be submitted immediately.
- An RFM customer table (`data.csv`, 616 purchasers) with frequency, monetary value, quartile ranks, and a combined RFM score.
- Notebook analysis of an e-commerce event log: event dictionary, brand coverage, views by weekday and hour, and the aggregation that produced the RFM table.
- A pickled `RandomForestRegressor` (`model.pkl`) trained to predict the `monetary` column.
- A separate RFMT customer table (`datamart_rfmt.csv`, 3,643 customers) and a file of 4-group labels (`cluster_labels_rfmt_k4.csv`).

## Analysis and model

### Event log and brand analysis

`Project2 copy.ipynb` (titled **Brand Analysis**) is the fuller write-up. It documents an event log whose columns are `event_time`, `event_type` (`view`, `cart`, `purchase`), `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, and `user_session`. Saved notebook outputs from that log describe a single day, 2019-10-01, and record:

- 46,047 users, 54,303 sessions, 23,874 products, 1,633 brands, and 3 event types
- 96,471 view events, all on weekday 2 in that slice, with views also broken out by hour
- 92.54% of purchase events carrying a brand and 7.46% with no brand
- Samsung, Apple, and Xiaomi as the leading brands by purchase count
- 748 purchase rows rolled up to 616 customers

The notebook states a marketing hypothesis from the brand split: most of a marketing budget should go to branded products, and a product introduced without a brand name is much less likely to be bought.

The event-level file itself is not in the repository. Those figures are the outputs stored in the notebooks. `Project2.ipynb` is a shorter copy of the same analysis. `Untitled.ipynb` is empty.

### RFM features

For each purchasing user the notebook computes:

| Feature | Definition in the notebook |
| --- | --- |
| `recency` | Time from the customer's last purchase to the latest purchase in the slice |
| `frequency` | Count of that customer's purchase rows |
| `monetary` | Sum of purchase `price` |
| `F_rank`, `M_rank` | Quartile ranks from 1 (highest) to 4 (lowest) |
| `RFM_Score` | `F_rank` and `M_rank` concatenated, so rank 4 and rank 2 become `42` |

A recency-rank function is defined in the notebook; the line that would apply it is commented out, and `data.csv` has no recency rank. On this slice the frequency quartiles all sit at 1, so `F_rank` takes only the values 1 and 4. The saved `describe()` output for the 616 customers reports frequency mean 1.21 (maximum 26) and monetary mean 347.12 (minimum 4.25, maximum 12,066.53). Monetary quartile cuts recorded in the notebook are 81.66, 174.76, and 388.68.

`data.csv` is that 616-row table. `model.py` reads it.

### Regressor

`model.py` fits a `RandomForestRegressor` with no constructor arguments:

- Features: `frequency`, `F_rank`, `M_rank`, `RFM_Score`
- Target: `monetary`
- Split: 70/30 (`test_size=0.3`, `random_state=50`)
- `StandardScaler` is imported; the scaling block is commented out, so the forest is fit on the raw numbers
- The fitted model is written to `model.pkl`

`model.pkl` records scikit-learn 0.24.1. `M_rank` is a quartile of `monetary`, and `RFM_Score` contains that rank, so two of the four inputs are computed from the target. The training script does not score the holdout split, and the repository stores no accuracy or error figure.

The page calls the prediction "lifetime monetary value" and labels the units as cedis. The training target is the `monetary` sum described above.

### RFMT table and cluster labels

`datamart_rfmt.csv` is a second customer table: `CustomerID`, `Recency`, `Frequency`, `MonetaryValue`, and `Tenure` for 3,643 customers. In the file, recency runs from 1 to 365, frequency from 1 to 1,497, monetary value from 0.65 to 48,060.35, and tenure from 2 to 365.

`cluster_labels_rfmt_k4.csv` is a single column of the integers 0, 1, 2, and 3, 3,643 lines, the same length as the datamart. The repository has no script that fits those groups. `model.py` and `app.py` do not read either file. The first code cell of `Project2 copy.ipynb` currently loads `datamart_rfmt.csv`; the output saved on that cell is the event log, so the cell source and the stored output disagree.

## Tech stack

- Python, with the notebooks last executed on Python 3.8.8
- pandas and NumPy in the notebooks, pandas in `model.py`, and NumPy in `app.py`
- Matplotlib for the weekday and hour charts in the notebooks
- scikit-learn 0.24.1 (`RandomForestRegressor`, `train_test_split`; `StandardScaler` imported only)
- pickle for `model.pkl`
- Flask, with a Jinja template in `templates/index.html`
- Bootstrap 5.2.0-beta1 and Google Fonts, loaded from CDNs in the template

Dependencies are not pinned in a requirements file.

## Repository layout

```
app.py                         Flask app: form, prediction, sample customer
model.py                       Train the forest and write model.pkl
model.pkl                      Serialized RandomForestRegressor
data.csv                       616-customer RFM table used for training
datamart_rfmt.csv              3,643-customer RFMT table
cluster_labels_rfmt_k4.csv     Integers 0–3, same row count as the datamart
Project2 copy.ipynb            Brand analysis, data dictionary, RFM build
Project2.ipynb                 Shorter copy of the same analysis
Untitled.ipynb                 Empty notebook
templates/index.html           Bootstrap form
readme.txt                     Short note that the app runs on localhost
```

`app.py` keeps the model in memory, loaded once at startup. `POST /predict` casts every submitted form value to float, in page order, and calls `model.predict`. `POST /generate` does not call the model. It draws a first name and a surname from two short lists (the lists include Simiel and Mensah) and fills number of purchases from 5–20, age from 18–70, gender as 1 or 2, and category from 1–4.

The form labels those four inputs as number of purchases, age, gender (1 = female, 2 = male), and category. The forest was trained on `frequency`, `F_rank`, `M_rank`, and `RFM_Score`. The predict route uses field order, so a score in the training feature space means entering those four columns in that order.

## Getting started

Prerequisites: Python 3 and pip. The notebooks record Python 3.8.8, and `model.pkl` records scikit-learn 0.24.1. Loading that pickle is most reliable with scikit-learn 0.24.1.

The app reads no environment variables and expects `model.pkl` in the working directory. No secret values are required.

```bash
pip install flask numpy "scikit-learn==0.24.1" pandas matplotlib
python app.py
```

`python app.py` starts Flask's development server with `debug=True`, which serves the form at [http://127.0.0.1:5000](http://127.0.0.1:5000). Submit the form to score the four fields, or use **Get User** to prefill a sample customer and submit that.

Retrain from the committed RFM table and overwrite `model.pkl`:

```bash
python model.py
```

The script prints the shape of `data.csv` and writes a new pickle. It does not print a test score.

There is no test runner and no build step in the repository. Re-running the brand-analysis cells needs the original event log, which is not committed. `data.csv` is already the RFM output of that analysis, and it is sufficient for `model.py`.

## Project status

Prototype. The analysis is a notebook named Project 2 and titled Brand Analysis, with a local Flask front end on the saved model. No course, capstone, or competition is named in the repository, and it publishes no leaderboard or deployment.

## What this demonstrates

- Turning a raw retail event stream into a customer table: session counts, spend, quartile ranks, and a composite RFM code, with the ranking rules written out explicitly.
- Exploratory analysis that states a hypothesis — branded products account for almost all purchases — and supports it with brand coverage and brand rankings, alongside weekday and hour view charts.
- A short train-and-serve loop in a few files: a held-out split, a scikit-learn random forest, pickle persistence, and a Flask route that scores form input with the loaded model.
- A small server-rendered UI, including a sample-customer generator, that makes the model something a person can operate in the browser.
