import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from mysql.connector import Error
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class StudentPerformanceAnalyzer:
    def __init__(self, host='localhost', user='root', password='your_password', database='student_analytics'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect_to_mysql(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                print("✓ Successfully connected to MySQL")
                return True
        except Error as e:
            print(f"✗ Error connecting to MySQL: {e}")
            return False
    
    def setup_database(self):
        cursor = self.connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cursor.execute(f"USE {self.database}")
        print(f"✓ Database '{self.database}' ready")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL,
                gender ENUM('Male', 'Female') NOT NULL,
                test_prep BOOLEAN NOT NULL,
                study_hours INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                score_id INT PRIMARY KEY AUTO_INCREMENT,
                student_id INT NOT NULL,
                subject VARCHAR(50) NOT NULL,
                score DECIMAL(5,2) NOT NULL,
                exam_date DATE NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                INDEX idx_student_subject (student_id, subject),
                INDEX idx_subject (subject)
            )
        """)
        
        self.connection.commit()
        print("✓ Tables created successfully")
        
    def generate_sample_data(self, num_students=200):
        cursor = self.connection.cursor()
        cursor.execute(f"USE {self.database}")
        cursor.execute("DELETE FROM scores")
        cursor.execute("DELETE FROM students")
        cursor.execute("ALTER TABLE students AUTO_INCREMENT = 1")
        
        subjects = ['Math', 'Science', 'English', 'History', 'Art']
        genders = ['Male', 'Female']
        
        print(f"Generating data for {num_students} students...")
        
        for i in range(num_students):
            gender = np.random.choice(genders)
            test_prep = bool(np.random.choice([0, 1]))
            study_hours = np.random.randint(5, 25)
            
            cursor.execute("""
                INSERT INTO students (name, gender, test_prep, study_hours)
                VALUES (%s, %s, %s, %s)
            """, (f"Student {i+1}", gender, test_prep, study_hours))
            
            student_id = cursor.lastrowid
            exam_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            for subject in subjects:
                base_score = np.random.uniform(60, 85)
                prep_bonus = 10 if test_prep else 0
                study_bonus = study_hours * 0.4
                subject_factor = {'Math': 0, 'Science': 2, 'English': 5, 'History': 3, 'Art': 8}
                
                score = min(100, base_score + prep_bonus + study_bonus + subject_factor.get(subject, 0))
                score = max(50, score + np.random.normal(0, 5))
                
                cursor.execute("""
                    INSERT INTO scores (student_id, subject, score, exam_date)
                    VALUES (%s, %s, %s, %s)
                """, (student_id, subject, round(score, 2), exam_date.date()))
        
        self.connection.commit()
        print(f"✓ Generated {num_students} students with {num_students * len(subjects)} score records")
    
    def analyze_subject_difficulty(self):
        query = """
            SELECT 
                subject,
                ROUND(AVG(score), 2) as average_score,
                ROUND(MIN(score), 2) as min_score,
                ROUND(MAX(score), 2) as max_score,
                ROUND(STDDEV(score), 2) as std_dev,
                COUNT(*) as total_students
            FROM scores
            GROUP BY subject
            ORDER BY average_score DESC
        """
        
        df = pd.read_sql(query, self.connection)
        
        print("\n" + "="*60)
        print("SUBJECT DIFFICULTY ANALYSIS")
        print("="*60)
        print(df.to_string(index=False))
        
        plt.figure(figsize=(12, 6))
        colors = ['#10b981' if x >= 85 else '#3b82f6' if x >= 75 else '#f59e0b' if x >= 65 else '#ef4444' 
                  for x in df['average_score']]
        bars = plt.bar(df['subject'], df['average_score'], color=colors, alpha=0.8, edgecolor='black')
        plt.axhline(y=df['average_score'].mean(), color='red', linestyle='--', label=f"Overall Avg: {df['average_score'].mean():.2f}")
        plt.xlabel('Subject', fontsize=12, fontweight='bold')
        plt.ylabel('Average Score', fontsize=12, fontweight='bold')
        plt.title('Subject Difficulty Analysis (Higher = Easier)', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('subject_difficulty.png', dpi=300, bbox_inches='tight')
        print("\n✓ Chart saved: subject_difficulty.png")
        plt.show()
        
        return df
    
    def analyze_gender_performance(self):
        query = """
            SELECT 
                s.gender,
                sc.subject,
                ROUND(AVG(sc.score), 2) as average_score,
                COUNT(*) as student_count
            FROM students s
            JOIN scores sc ON s.student_id = sc.student_id
            GROUP BY s.gender, sc.subject
            ORDER BY s.gender, sc.subject
        """
        
        df = pd.read_sql(query, self.connection)
        pivot_df = df.pivot(index='subject', columns='gender', values='average_score')
        
        print("\n" + "="*60)
        print("GENDER-BASED PERFORMANCE COMPARISON")
        print("="*60)
        print(pivot_df.to_string())
        
        pivot_df['Difference (M-F)'] = pivot_df['Male'] - pivot_df['Female']
        print(f"\nGender Performance Gap:\n{pivot_df['Difference (M-F)'].to_string()}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(pivot_df.index))
        width = 0.35
        ax1.bar(x - width/2, pivot_df['Male'], width, label='Male', color='#3b82f6', alpha=0.8)
        ax1.bar(x + width/2, pivot_df['Female'], width, label='Female', color='#ec4899', alpha=0.8)
        ax1.set_xlabel('Subject', fontweight='bold')
        ax1.set_ylabel('Average Score', fontweight='bold')
        ax1.set_title('Gender Performance by Subject', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pivot_df.index)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        sns.heatmap(pivot_df[['Male', 'Female']], annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=60, vmax=90, ax=ax2, cbar_kws={'label': 'Average Score'})
        ax2.set_title('Gender Performance Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('gender_performance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Chart saved: gender_performance.png")
        plt.show()
        
        return df
    
    def analyze_test_prep_effect(self):
        query = """
            SELECT 
                sc.subject,
                ROUND(AVG(CASE WHEN s.test_prep = TRUE THEN sc.score END), 2) as avg_with_prep,
                ROUND(AVG(CASE WHEN s.test_prep = FALSE THEN sc.score END), 2) as avg_without_prep,
                ROUND(AVG(CASE WHEN s.test_prep = TRUE THEN sc.score END) - 
                      AVG(CASE WHEN s.test_prep = FALSE THEN sc.score END), 2) as improvement
            FROM students s
            JOIN scores sc ON s.student_id = sc.student_id
            GROUP BY sc.subject
            ORDER BY improvement DESC
        """
        
        df = pd.read_sql(query, self.connection)
        
        print("\n" + "="*60)
        print("TEST PREPARATION IMPACT ANALYSIS")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nAverage Improvement with Test Prep: {df['improvement'].mean():.2f} points")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(df['subject'], df['avg_with_prep'], marker='o', linewidth=2, 
                markersize=8, label='With Test Prep', color='#10b981')
        ax1.plot(df['subject'], df['avg_without_prep'], marker='s', linewidth=2, 
                markersize=8, label='Without Test Prep', color='#ef4444')
        ax1.set_xlabel('Subject', fontweight='bold')
        ax1.set_ylabel('Average Score', fontweight='bold')
        ax1.set_title('Test Prep Effect Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(60, 100)
        
        colors = ['#10b981' if x > df['improvement'].mean() else '#f59e0b' for x in df['improvement']]
        bars = ax2.bar(df['subject'], df['improvement'], color=colors, alpha=0.8, edgecolor='black')
        ax2.axhline(y=df['improvement'].mean(), color='red', linestyle='--', 
                   label=f"Avg Improvement: {df['improvement'].mean():.2f}")
        ax2.set_xlabel('Subject', fontweight='bold')
        ax2.set_ylabel('Score Improvement', fontweight='bold')
        ax2.set_title('Test Prep Improvement by Subject', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('test_prep_effect.png', dpi=300, bbox_inches='tight')
        print("\n✓ Chart saved: test_prep_effect.png")
        plt.show()
        
        return df
    
    def analyze_study_hours_correlation(self):
        query = """
            SELECT 
                s.study_hours,
                ROUND(AVG(sc.score), 2) as average_score,
                COUNT(DISTINCT s.student_id) as student_count
            FROM students s
            JOIN scores sc ON s.student_id = sc.student_id
            GROUP BY s.study_hours
            ORDER BY s.study_hours
        """
        
        df = pd.read_sql(query, self.connection)
        correlation = df['study_hours'].corr(df['average_score'])
        
        print("\n" + "="*60)
        print("STUDY HOURS CORRELATION ANALYSIS")
        print("="*60)
        print(f"Correlation coefficient: {correlation:.3f}")
        print(f"\nStudy Hours vs Average Score:")
        print(df.to_string(index=False))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.scatter(df['study_hours'], df['average_score'], s=100, alpha=0.6, color='#8b5cf6')
        z = np.polyfit(df['study_hours'], df['average_score'], 1)
        p = np.poly1d(z)
        ax1.plot(df['study_hours'], p(df['study_hours']), "r--", linewidth=2, 
                label=f'Trend Line (r={correlation:.3f})')
        ax1.set_xlabel('Weekly Study Hours', fontweight='bold')
        ax1.set_ylabel('Average Score', fontweight='bold')
        ax1.set_title('Study Hours vs Performance', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        query2 = """
            SELECT 
                CASE 
                    WHEN s.study_hours < 10 THEN '5-9 hrs'
                    WHEN s.study_hours < 15 THEN '10-14 hrs'
                    WHEN s.study_hours < 20 THEN '15-19 hrs'
                    ELSE '20+ hrs'
                END as study_range,
                sc.score
            FROM students s
            JOIN scores sc ON s.student_id = sc.student_id
        """
        df2 = pd.read_sql(query2, self.connection)
        sns.boxplot(data=df2, x='study_range', y='score', ax=ax2, palette='Set2')
        ax2.set_xlabel('Study Hours Range', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Score Distribution by Study Hours', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('study_hours_correlation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Chart saved: study_hours_correlation.png")
        plt.show()
        
        return df
    
    def generate_comprehensive_report(self):
        query = """
            SELECT 
                COUNT(DISTINCT s.student_id) as total_students,
                ROUND(AVG(sc.score), 2) as overall_average,
                SUM(CASE WHEN s.test_prep = TRUE THEN 1 ELSE 0 END) as students_with_prep,
                SUM(CASE WHEN s.test_prep = FALSE THEN 1 ELSE 0 END) as students_without_prep,
                SUM(CASE WHEN s.gender = 'Male' THEN 1 ELSE 0 END) as male_students,
                SUM(CASE WHEN s.gender = 'Female' THEN 1 ELSE 0 END) as female_students,
                ROUND(AVG(s.study_hours), 2) as avg_study_hours
            FROM students s
            JOIN scores sc ON s.student_id = sc.student_id
        """
        
        df = pd.read_sql(query, self.connection)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Students: {df['total_students'].iloc[0]}")
        print(f"Overall Average Score: {df['overall_average'].iloc[0]}")
        print(f"Students with Test Prep: {df['students_with_prep'].iloc[0]}")
        print(f"Students without Test Prep: {df['students_without_prep'].iloc[0]}")
        print(f"Male Students: {df['male_students'].iloc[0]}")
        print(f"Female Students: {df['female_students'].iloc[0]}")
        print(f"Average Study Hours: {df['avg_study_hours'].iloc[0]}")
        print("="*60)
        
        return df
    
    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("\n✓ MySQL connection closed")


def main():
    print("\n" + "="*60)
    print("STUDENT ACADEMIC PERFORMANCE ANALYSIS SYSTEM")
    print("Using: pandas, matplotlib, seaborn, MySQL")
    print("="*60 + "\n")
    
    # UPDATE THESE CREDENTIALS FOR YOUR MySQL SERVER
    analyzer = StudentPerformanceAnalyzer(
        host='localhost',           # Your MySQL host (usually 'localhost')
        user='root',                # Your MySQL username
        password='YOUR_PASSWORD',   # Replace with your actual MySQL password
        database='student_analytics'
    )
    
    if not analyzer.connect_to_mysql():
        print("\nPlease update MySQL credentials in the code and try again.")
        return
    
    analyzer.setup_database()
    analyzer.generate_sample_data(num_students=200)
    
    print("\n" + "🔍 Starting comprehensive analysis...\n")
    
    analyzer.analyze_subject_difficulty()
    analyzer.analyze_gender_performance()
    analyzer.analyze_test_prep_effect()
    analyzer.analyze_study_hours_correlation()
    analyzer.generate_comprehensive_report()
    
    analyzer.close_connection()
    
    print("\n✅ Analysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
