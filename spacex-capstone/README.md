# SpaceX Falcon 9 Landing Analysis

IBM Applied Data Science Capstone. Prepared for Dakshesh Gautam with AI assistance.

This project analyzes historical launch and landing records, compares four classifiers, and includes a Folium map and a Plotly Dash dashboard. The PDF report covers all final-project rubric sections. Every reported statistic comes from executed Python or SQL code.

## Reproduce

```bash
python -m pip install -r requirements.txt
python download_data.py
python analysis.py
python dashboard.py
```

Open http://127.0.0.1:8050 for the dashboard. Open `launch_sites_map.html` and `site_proximity_map.html` for interactive maps. Run notebooks from the project root or the `notebooks` directory. All notebooks contain executed code and actual outputs. EDA images, SQL query outputs, model results and the SQLite database are generated in `results/`.

## Notebooks

1. `01_data_collection_api.ipynb`: parses the official IBM historical SpaceX API response.
2. `02_web_scraping.ipynb`: uses BeautifulSoup to extract historical Wikipedia launch-table records.
3. `03_data_wrangling.ipynb`: prepares the landing target and validates it against the course checkpoint.
4. `04_eda_visualization.ipynb`: produces all six required plots and grouped summaries.
5. `05_eda_sql.ipynb`: executes all ten required SQL queries in SQLite.
6. `06_interactive_folium.ipynb`: maps launch outcomes and computes geographic proximity.
7. `07_predictive_analysis.ipynb`: tunes logistic regression, SVM, decision tree and KNN.
8. `08_plotly_dash.ipynb`: summarizes dashboard outputs. `dashboard.py` implements the interactive application.

## Data provenance and definitions

The modeling data are IBM's published API-derived checkpoint, with 90 Falcon 9 launches from 2010-06-04 through 2020-11-05. The raw archived API response contains 107 launches across rocket types. The code parses this response but uses the course normalized checkpoint for modeling rather than claiming it freshly rebuilt all reference-table fields from the live API. Payload mass is already imputed in that checkpoint. Missing landing pad values remain missing and are excluded from prediction.

The target follows the course definition: `True ASDS`, `True RTLS` and `True Ocean` map to class 1. Failure and no-recovery records map to class 0. Thus class 0 combines failures with missions where recovery was not attempted. Successful ocean landings do not necessarily imply an economically reusable booster.

SQL uses the separate 101-row course `Spacex.csv` dataset. Mission success differs from booster landing success. Dashboard and map each use a 56-row course snapshot. Launch-site label conventions differ between datasets. They are intentionally analyzed separately.

The Wikipedia scrape contains 81 full launch-table rows for 2010-2019, including Falcon Heavy and a prelaunch loss. It is a separate collection demonstration. OpenStreetMap geometry comes from the official API. Proximity uses nearest mapped line segments and a local planar projection followed by haversine distance. Roads exclude service roads. Railway proximity includes a mapped launch-pad spur, so its near-zero distance reflects map geometry and the representative site coordinate, not a transportation claim.

## Prediction method

The fixed stratified split has 72 training and 18 test launches. Five-fold cross-validation tunes models on training data. Scaling and encoding fit inside the pipeline. Outcome and landing-pad fields are excluded. Numeric predictors are flight number, payload mass, booster flight count, block, reuse count, grid fins, reuse and legs. Categorical predictors are orbit and launch site.

SVM has the highest training cross-validation accuracy (87.5%). Logistic regression has the highest test accuracy (83.3%). The CV-selected SVM test accuracy is 72.2%, against a 66.7% majority baseline. A chronological SVM check scores 77.8%, below an 83.3% majority baseline for the recent sample. Results describe a small historical dataset and do not establish production performance or present-day launch economics.

## Sources

All IBM input URLs are listed in `download_data.py`.

- IBM course: https://www.coursera.org/learn/applied-data-science-capstone
- Historical launch list: https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches_%282010%E2%80%932019%29
- SpaceX API documentation: https://github.com/r-spacex/SpaceX-API
- OpenStreetMap: https://www.openstreetmap.org/copyright

No other learner's completed solution was used.
