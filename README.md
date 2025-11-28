Student Academic Performance Analysis

A Python-based data analysis system that uses MySQL to analyze student performance trends across multiple subjects and factors.

Project Overview

This project analyzes academic performance data for 200 students across 5 subjects (Math, Science, English, History, Art) to identify key trends and insights including:

- Subject difficulty rankings
- Gender-based performance comparisons
- Test preparation effectiveness
- Study hours correlation with grades

Technologies Used

- **Python 3.x**
- **MySQL** - Relational database for data storage
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations
- **numpy** - Numerical computations

Database Schema

### Students Table
- `student_id` (Primary Key)
- `name`
- `gender` (Male/Female)
- `test_prep` (Boolean)
- `study_hours` (Integer)
- `created_at` (Timestamp)

### Scores Table
- `score_id` (Primary Key)
- `student_id` (Foreign Key)
- `subject`
- `score` (Decimal)
- `exam_date` (Date)

## 🚀 Installation

1. Install required Python packages:
```bash
pip install pandas matplotlib seaborn mysql-connector-python numpy
```

2. Set up MySQL server and update credentials in code:
```python
analyzer = StudentPerformanceAnalyzer(
    host='localhost',
    user='root',
    password='YOUR_PASSWORD',
    database='student_analytics'
)
```

Usage

Run the script:
```bash
python student_analysis.py
```

Output

The script generates:
- Console output with detailed statistics
- 4 PNG visualization files:
  - `subject_difficulty.png` - Bar chart of average scores by subject
  - `gender_performance.png` - Gender comparison charts and heatmap
  - `test_prep_effect.png` - Impact of test preparation
  - `study_hours_correlation.png` - Study time vs performance analysis

Key Features

✅ Automated data generation with realistic patterns  
✅ Complex SQL queries with JOINs and aggregations  
✅ Statistical analysis (correlation, std deviation)  
✅ Professional visualizations with multiple chart types  
✅ Comprehensive performance reporting  

Sample Insights

- Identifies easiest/hardest subjects based on average scores
- Measures test prep improvement (typically +7-10 points)
- Shows positive correlation between study hours and performance
- Compares gender performance gaps across subjects

Use Cases

- Educational institutions analyzing student performance
- Identifying subjects requiring additional support
- Evaluating effectiveness of test preparation programs
- Data-driven curriculum decisions

Author

SMARAK11323

License

This project is open source and available for educational purposes.
