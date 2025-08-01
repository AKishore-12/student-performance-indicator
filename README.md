## 🧠 Student Performance Indicator

A machine learning project that predicts a student's math score based on demographic and academic information. It features an interactive web interface and supports MLflow for experiment tracking.

📊 Dataset

The project uses the stud.csv dataset, which contains the following features:
| Feature                       | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `gender`                      | Student's gender (`male`, `female`)                  |
| `race_ethnicity`              | Student's group (`group A` to `group E`)             |
| `parental_level_of_education` | Highest education level of the student's parent(s)   |
| `lunch`                       | Type of lunch (`standard`, `free/reduced`)           |
| `test_preparation_course`     | Test preparation course status (`none`, `completed`) |
| `reading_score`               | Student's reading score (0-100)                      |
| `writing_score`               | Student's writing score (0-100)                      |
| `math_score`                  | **Target variable** — math score (0-100)             |


🗂️ Project Structure

.
├── app.py                     # Flask backend
├── templates/                # HTML frontend templates
├── src/                      # Source code (pipeline, training, prediction)
│   ├── components/           # Data ingestion, transformation, and model training logic
│   ├── pipeline/             # Training and prediction pipelines
│   ├── utils.py              # Utility functions
│   └── exception.py          # Custom exceptions
├── artifacts/                # Stored models and preprocessors
├── src/notebook/             # Jupyter notebooks for EDA and experiments
├── requirements.txt          # Python dependencies
├── Dockerfile                # For containerization
└── README.md                 # Project documentation

⚙️ How It Works

• Users fill in student details via the web form.

• The backend processes the input and uses a trained ML model to predict the math score.

• The result is displayed on the webpage.

🚀 Getting Started

1. Clone the Repository
git clone <repo-url>
cd student-performance-indicator

2. Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Start the Application
python app.py
Visit: http://localhost:8000

🧪 MLflow Experiment Tracking

This project integrates MLflow to track model training runs, hyperparameters, and performance metrics.

🔍 What MLflow Tracks
• Model names and algorithms

• Hyperparameters from GridSearchCV

• R² scores for each model

• Saved best model artifact

📌 How to Use MLflow

1. Start MLflow UI
mlflow ui --port 8000
Then go to http://localhost:8000 to view experiments.

2. Train the Model
Training the model (via script or notebook) will automatically log:

• Parameters

• Metrics

• Models under the experiment name: performance_indicator

3. Compare Runs
• Use the MLflow UI to:

• Compare performance across models

• Download or register the best model

• Track experiments over time

🐳 Docker Support (Optional)

To build and run the app with Docker:
docker build -t student-performance .
docker run -p 8000:8000 student-performance

🖥️ Frontend Usage

• Open the prediction page in the browser.

• Fill in all required fields:

    Gender, race/ethnicity, parental education level, etc.

• Click "Predict your Maths Score"

• View the predicted score instantly!

📈 Model Training & Notebooks

• For in-depth data exploration and training logic, refer to:

• src/notebook/EDA Student Performance.ipynb

• src/notebook/Model Training.ipynb

📝 License

This project is intended for educational and learning purposes only.

