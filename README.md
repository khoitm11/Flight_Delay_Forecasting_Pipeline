# Flight Delay Prediction ✈️⏰

Hey everyone! This project is all about building a system to predict flight delay rates. I took on the whole journey: from grabbing raw data from the Bureau of Transportation Statistics (BTS), cleaning it up, training a few different machine learning models with Scikit-learn, keeping track of all my experiments using MLflow, and finally, packing the whole thing into a neat Docker container so anyone can run it!

**Quick Heads-up on Model Predictions:**
The main goal here was to build a complete end-to-end pipeline and learn the ropes. The models trained are baseline regressors. For super-accurate, ready-for-production predictions, the next steps would definitely involve more in-depth feature engineering, fine-tuning model settings (hyperparameters), and maybe trying out more complex models.

## 🚀 What I Did (My Project Journey)

It was a step-by-step process:

1.  **Getting the Data:** I downloaded flight on-time performance data for Q4 2023 (October, November, December) directly from the [BTS website](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp). This involved figuring out their filters and getting the CSV files.

2.  **Data Cleaning & Prep (`notebooks/01_Data_Prep.ipynb`):**
    *   Loaded the three monthly CSVs into Pandas DataFrames and combined them.
    *   Cleaned it up: got rid of an "Unnamed" column that sometimes appears, handled missing values (NaNs) for key fields. For instance, if `arr_del15` (flights delayed 15+ mins) was NaN, I assumed 0 delays for calculating the rate.
    *   **Created the Target Variable:** Engineered the `delay_rate` which is `arr_del15 / arr_flights`. I also made sure this rate stayed between 0 and 1.

3.  **Exploratory Data Analysis (EDA) & More Features (Often in `notebooks/02_EDA_...ipynb` or integrated):**
    *   Took a look at how data was distributed, like `delay_rate` histograms.
    *   Created new features, for example, `carrier_ct_rate` (carrier-caused delay rate) by dividing `carrier_ct` (count of carrier-caused delays) by `arr_flights`. I did this for other delay causes too.
    *   **Oops & Learning Moment:** Initially, creating these `*_ct_rate` features *before* splitting data led to some "too good to be true" results with simpler models (like Linear Regression showing R² near 1!). Realized this was a form of data leakage. For a production system, I'd be more careful to create these features post-split or using only training data information. For this project, I kept them to demonstrate the full pipeline with these features.

4.  **Getting Ready for Modeling (`notebooks/03_Feature_Engineering_Modeling_Prep.ipynb`):**
    *   Picked the final set of features I thought would be useful (e.g., `month`, `carrier`, `airport`, `arr_flights`, and the `*_ct_rate`s).
    *   Split the data: 80% for training the models, 20% for testing how well they learned.
    *   **Preprocessed the Features:** Used Scikit-learn's `ColumnTransformer`:
        *   `StandardScaler` for numeric features (to get them on a similar scale).
        *   `OneHotEncoder` for categorical features (like `carrier` and `airport` codes) to turn them into numbers.
    *   Saved the fitted `ColumnTransformer` (as `data_preprocessor.joblib`) and a JSON file (`feature_config.json`) detailing the features and their order. This is super important for making predictions on new data later!

5.  **Training & Tracking Models (also in `notebooks/03_...`):**
    *   Set up **MLflow** to track everything. I gave my experiment a name like `flight_delay_project_Model_Training`.
    *   Wrote a helper function to train a model, evaluate it, and log everything to MLflow. This included:
        *   Model parameters.
        *   Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² Score.
        *   The model itself (as an MLflow artifact).
        *   The preprocessor and feature config.
        *   Some plots like "Actual vs. Predicted" and feature importances (for tree-based models).
    *   Trained several models: Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
    *   Picked the "best" model (Gradient Boosting Regressor often performed well here, especially after being cautious about the earlier data leakage suspicion) based on the metrics in the MLflow UI. Saved this model as `final_best_flight_delay_model.joblib`.

6.  **Building the Prediction App (`app/app.py`):**
    *   Wrote a Python script that uses `argparse` to take flight details as command-line arguments (like airport, carrier, number of flights, etc.).
    *   This script loads the saved best model, the preprocessor, and the feature config.
    *   It then prepares the input data in the same way the training data was prepared and makes a prediction for the `delay_rate`.

7.  **Dockerizing It! (`Dockerfile`):**
    *   Wrote a `Dockerfile` to create a portable environment for the app.
        *   Started from a Python base image (e.g., `python:3.9-slim-bullseye`).
        *   Copied over the `requirements.txt` and installed all needed libraries.
        *   Copied the `app` folder (with `app.py`) and the `saved_models` folder.
        *   Set the `CMD` to run `app/app.py` with some default example values.
    *   **Problem Solving Time:**
        *   Hit a `ModuleNotFoundError: No module named '_loss'` when running the Docker container. This was because my local Python (e.g., 3.12 where I trained) had a newer Scikit-learn than what got installed in Docker (initially trying Python 3.8).
        *   **Solution:** Upgraded the Python version in the `Dockerfile` to something like 3.9 or 3.10 and made sure my `requirements.txt` specified Scikit-learn versions compatible with *that* Python, leading to a consistent environment.
        *   Used a `.dockerignore` file to keep the Docker build context small and fast.
    *   Built the Docker image: `docker build -t flight-delay-predictor:v3 .`
    *   Successfully ran the container using the command you see below to make predictions!

## 🛠️ Tech Stack I Used

*   **Python:** Version 3.9+ (consistent with the Docker image)
*   **Core Data Libraries:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Model Saving/Loading:** Joblib
*   **Visualization:** Matplotlib, Seaborn (mostly in notebooks)
*   **Experiment Tracking:** MLflow
*   **Containerization:** Docker
*   **IDE & Notebooks:** PyCharm (or VS Code) with Jupyter
*   **Version Control:** Git & GitHub

## 📊 The Data

*   **Source:** U.S. Department of Transportation - Bureau of Transportation Statistics (BTS)
*   **Specific Dataset:** On-Time Performance & Delay Causes
*   **Link:** [https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)
*   **Time Period I Used:** Q4 2023 (October, November, December)
*   *(If you want to run the data prep notebooks, you'll need to download these CSVs yourself and put them in the `data/bts_raw/` folder.)*

## ⚙️ What You'll Need to Run This

*   Python (3.9 or higher recommended for consistency)
*   Pip (Python's package installer)
*   Git (for cloning the project)
*   Docker Desktop (make sure it's running!)

## 🏁 Getting Started & Using The Project

1.  **Clone This Project:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT_REPO_NAME.git # <-- IMPORTANT: Change this URL!
    cd YOUR_PROJECT_REPO_NAME
    ```

2.  **Set Up Data (If you want to re-run data prep/training notebooks):**
    *   Go to the BTS link above and download the CSV files for October, November, and December 2023 (Reporting Carrier On-Time Performance).
    *   Extract them and place the CSV files into the `data/bts_raw/` directory within this project.

3.  **Set Up a Python Virtual Environment (Good practice!):**
    ```bash
    python -m venv venv
    # On Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Required Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```

### A. Running the Jupyter Notebooks (For Data Prep & Model Training)

1.  Open the project folder in your favorite IDE (like PyCharm or VS Code).
2.  Make sure your IDE is using the Python interpreter from the `venv` you created.
3.  Run the notebooks, usually in this order:
    *   `notebooks/01_Data_Prep.ipynb`
    *   `notebooks/02_EDA_and_Feature_Enrichment.ipynb` (if you have a separate one for EDA)
    *   `notebooks/03_Feature_Engineering_Modeling_Prep.ipynb` (This one does the training and saves models!)
4.  **Check out MLflow Experiments:**
    After running the training notebook (03), open a new terminal in the project's root directory and type:
    ```bash
    mlflow ui
    ```
    Then, open your web browser and go to `http://localhost:5000`. You should see your experiment and all the logged runs!

### B. Running the Prediction App (`app/app.py`)

**1. Locally (After training and saving models):**

   Make sure `final_best_flight_delay_model.joblib`, `data_preprocessor.joblib`, and `feature_config.json` are in the `saved_models/` directory.

   From the project's root directory, you can run the script like this (example for PowerShell, adjust for your shell):
   ```powershell
   python app/app.py --airport "JFK" --arr_cancelled "0" --arr_diverted "0" --arr_flights "120" --carrier "AA" --carrier_ct_rate "0.1" --late_aircraft_ct_rate "0.05" --month "10" --nas_ct_rate "0.02" --security_ct_rate "0.0" --weather_ct_rate "0.01"
   ```
2. Using Docker 🐳 (The recommended way to run the packaged app!)

a. Build the Docker Image:
```
docker build -t flight-delay-predictor:v3 .
```
b. Run the Docker Container to Make a Prediction:

# For PowerShell (using backticks ` for line continuation)
```
docker run --rm flight-delay-predictor:v3 python app/app.py `
  --airport "JFK" `
  --arr_cancelled "0" `
  --arr_diverted "0" `
  --arr_flights "120" `
  --carrier "AA" `
  --carrier_ct_rate "0.1" `
  --late_aircraft_ct_rate "0.05" `
  --month "10" `
  --nas_ct_rate "0.02" `
  --security_ct_rate "0.0" `
  --weather_ct_rate "0.01"
```

# For Bash (Linux/macOS, using backslash \ for line continuation)
```
 docker run --rm flight-delay-predictor:v3 python app/app.py
   --airport "JFK" \
   --arr_cancelled "0" \
   --arr_diverted "0" \
   --arr_flights "120" \
   --carrier "AA" \
   --carrier_ct_rate "0.1" \
   --late_aircraft_ct_rate "0.05" \
   --month "10" \
   --nas_ct_rate "0.02" \
   --security_ct_rate "0.0" \
   --weather_ct_rate "0.01"
```
The predicted delay rate will be printed to your console!

## 📂 Project Structure

-   `YOUR_PROJECT_REPO_NAME/`
    -   `.git/` *(Git repository data - usually hidden, ignored by user)*
    -   `.idea/` *(PyCharm project files - should be in .gitignore)*
    -   `venv/` *(Python virtual environment - should be in .gitignore)*
    -   `app/` *(Main prediction application code)*
        -   `app.py` *(The command-line prediction script)*
    -   `data/` *(Holds datasets)*
        -   `bts_raw/` *(Place raw downloaded CSVs here - should be in .gitignore)*
        -   `bts_processed/` *(Output from data prep notebook - should be in .gitignore)*
    -   `mlruns/` *(MLflow data for local runs - should be in .gitignore)*
    -   `notebooks/` *(Jupyter notebooks for development)*
        -   `01_Data_Prep.ipynb`
        -   `02_EDA_and_Feature_Enrichment.ipynb` *(Or similar name)*
        -   `03_Feature_Engineering_Modeling_Prep.ipynb`
    -   `saved_models/` *(Stores trained model, preprocessor, feature config)*
    -   `visualizations/` *(Stores plots - if also in MLflow, could be in .gitignore)*
    -   `.dockerignore` *(Files to ignore for Docker build context)*
    -   `.gitignore` *(Files/directories Git should ignore)*
    -   `Dockerfile` *(Instructions to build the Docker image)*
    -   `requirements.txt` *(Python package dependencies)*
    -   `README.md` *(This file!)*


👨‍💻 About Me

Name: Khoi Tran

LinkedIn: www.linkedin.com/in/khoitm11
