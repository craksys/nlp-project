import numpy as np
from scipy import stats
import pandas as pd
from svm import SVMClassifier
from bayes import NaiveBayesClassifier
from bert import BERTClassifier
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class ModelComparison:
    def __init__(self, data_path="tripadvisor_hotel_reviews.csv", n_splits=4):
        self.svm = SVMClassifier(data_path)
        self.bayes = NaiveBayesClassifier(data_path)
        self.bert = BERTClassifier(data_path)
        self.n_splits = n_splits
        self.modes = ['1_5', 'with_neutral', 'without_neutral']
        self.models = ['SVM', 'Naive Bayes', 'BERT']
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    def prepare_data(self, mode='1_5'):
        """Prepare data for all models"""
        print("\nPreprocessing reviews...")
        self.svm.df['Processed_Review'] = self.svm.df['Review'].apply(self.svm.preprocess_text)
        print("Preprocessing complete.")
        print("Example of processed review:\n", self.svm.df[['Review', 'Processed_Review']].head())
        
        self.svm.df['Sentiment'] = self.svm.df['Rating'].apply(
            lambda x: self.svm.categorize_rating(x, mode=mode))
        print("\nRating distribution:\n", self.svm.df['Sentiment'].value_counts().sort_index())
        
        return self.svm.df['Processed_Review'], self.svm.df['Sentiment']
    
    def preprocess_text_with_features(self, text, features):
        """Preprocess text with specified feature combinations"""
        if features['lowercase']:
            text = text.lower()
        
        if features['remove_special_chars']:
            text = re.sub(r'[^a-z0-9\s]', '', text)
        
        tokens = word_tokenize(text)
        
        if features['remove_stopwords']:
            tokens = [word for word in tokens if word not in self.svm.stop_words]
        
        if features['lemmatize']:
            tokens = [self.svm.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def run_cross_validation(self, mode='1_5', random_state=42):
        """Run k-fold cross validation for all models"""
        print(f"\nRunning {self.n_splits}-fold cross validation for mode: {mode}")
        print("=" * 50)
        
        cv_results = {model: {metric: [] for metric in self.metrics} for model in self.models}
        
        X, y = self.prepare_data(mode=mode)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)

        X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        print("\nTraining BERT model (single split)...")
        bert_model, bert_tokenizer, bert_metrics, _, _ = self.bert.train_and_evaluate(
            mode=mode, X_train=X_train_bert, X_test=X_test_bert, 
            y_train=y_train_bert, y_test=y_test_bert)
        for metric in self.metrics:
            cv_results['BERT'][metric] = [bert_metrics[metric]] * self.n_splits
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{self.n_splits}")
            print("-" * 30)
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            svm_model, svm_vectorizer, svm_metrics, _, _ = self.svm.train_and_evaluate(
                mode=mode, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            for metric in self.metrics:
                cv_results['SVM'][metric].append(svm_metrics[metric])
            
            nb_model, nb_vectorizer, nb_metrics, _, _ = self.bayes.train_and_evaluate(
                mode=mode, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            for metric in self.metrics:
                cv_results['Naive Bayes'][metric].append(nb_metrics[metric])
        
        return cv_results
    
    def perform_wilcoxon_tests(self, cv_results):
        """Perform Wilcoxon signed-rank tests between models"""
        significance_matrix = {metric: np.zeros((len(self.models), len(self.models))) for metric in self.metrics}
        
        print("\nWilcoxon Signed-Rank Test Results:")
        print("=" * 50)
        
        for metric in self.metrics:
            print(f"\n{metric.upper()} Comparison:")
            for i, model1 in enumerate(self.models):
                for j, model2 in enumerate(self.models):
                    if i != j:
                        scores1 = cv_results[model1][metric]
                        scores2 = cv_results[model2][metric]
                        
                        statistic, p_value = stats.wilcoxon(scores1, scores2)
                        
                        if p_value < 0.05:
                            if np.mean(scores1) > np.mean(scores2):
                                significance_matrix[metric][i, j] = 1
                            else:
                                significance_matrix[metric][i, j] = -1
                        
                        print(f"\n{model1} vs {model2}:")
                        print(f"p-value: {p_value:.4f}")
                        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                        print(f"Mean {model1}: {np.mean(scores1):.4f}")
                        print(f"Mean {model2}: {np.mean(scores2):.4f}")
                        print("-" * 30)
        
        return significance_matrix
    
    def perform_mode_comparison_tests(self, all_results):
        """Perform statistical tests between different modes for each model"""
        print("\nMode Comparison Results:")
        print("=" * 50)
        
        for model in self.models:
            print(f"\n{model} Mode Comparison:")
            for metric in self.metrics:
                print(f"\n{metric.upper()}:")
                data = [all_results[mode][model][metric] for mode in self.modes]
                statistic, p_value = stats.friedmanchisquare(*data)
                
                print(f"Friedman test p-value: {p_value:.4f}")
                print("Significant difference between modes:", "Yes" if p_value < 0.05 else "No")
                
                for mode in self.modes:
                    mean_value = np.mean(all_results[mode][model][metric])
                    std_value = np.std(all_results[mode][model][metric])
                    print(f"{mode}: {mean_value:.4f} ± {std_value:.4f}")
                print("-" * 30)
    
    def plot_metrics_by_mode(self, all_results):
        """Create plots showing metrics across different modes for each model"""
        plt.style.use('seaborn')
        
        for model in self.models:
            plt.figure(figsize=(12, 6))
            
            data = []
            for mode in self.modes:
                for metric in self.metrics:
                    mean_value = np.mean(all_results[mode][model][metric])
                    data.append({
                        'Mode': mode,
                        'Metric': metric,
                        'Value': mean_value
                    })
            
            df = pd.DataFrame(data)
            
            sns.lineplot(data=df, x='Mode', y='Value', hue='Metric', marker='o')
            plt.title(f'Metrics Across Modes - {model}')
            plt.xlabel('Classification Mode')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.tight_layout()
            
            plt.show()
            plt.close()
    
    def print_significance_matrix(self, significance_matrix, mode):
        """Print the significance matrix in a clear format"""
        print(f"\nSignificance Matrix for {mode} mode:")
        print("=" * 50)
        print("Legend: 1 = row model significantly better than column model")
        print("       -1 = row model significantly worse than column model")
        print("        0 = no significant difference")
        print("=" * 50)
        
        for metric in significance_matrix.keys():
            print(f"\n{metric.upper()}:")
            print("-" * 50)
            print("Model".ljust(15), end="")
            for model in self.models:
                print(f"{model}".ljust(15), end="")
            print("\n" + "-" * 60)
            
            for i, model1 in enumerate(self.models):
                print(f"{model1}".ljust(15), end="")
                for j, model2 in enumerate(self.models):
                    if i == j:
                        print("N/A".ljust(15), end="")
                    else:
                        value = significance_matrix[metric][i, j]
                        if value == 1:
                            print("1 (better)".ljust(15), end="")
                        elif value == -1:
                            print("-1 (worse)".ljust(15), end="")
                        else:
                            print("0 (ns)".ljust(15), end="")
                print()
            print()
    
    def run_model_comparison_experiment(self):
        """Experiment 1: Compare models within each mode"""
        print("\nExperiment 1: Model Comparison within Modes")
        print("=" * 50)
        
        all_results = {}
        
        for mode in self.modes:
            print(f"\nRunning experiments for mode: {mode}")
            print("=" * 50)
            
            cv_results = self.run_cross_validation(mode=mode)
            all_results[mode] = cv_results
            
            significance_matrix = self.perform_wilcoxon_tests(cv_results)
            
            self.print_significance_matrix(significance_matrix, mode)
            
            print("\nCross-validation Summary Statistics:")
            print("=" * 50)
            for model in self.models:
                print(f"\n{model}:")
                for metric in self.metrics:
                    scores = cv_results[model][metric]
                    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return all_results
    
    def run_mode_comparison_experiment(self, all_results):
        """Experiment 2: Compare different classification modes for each model"""
        print("\nExperiment 2: Mode Comparison for Each Model")
        print("=" * 50)
        
        for model in self.models:
            print(f"\n{model} Mode Comparison:")
            for metric in self.metrics:
                print(f"\n{metric.upper()}:")
                
                for i, mode1 in enumerate(self.modes):
                    for mode2 in self.modes[i+1:]:
                        scores1 = all_results[mode1][model][metric]
                        scores2 = all_results[mode2][model][metric]
                        
                        statistic, p_value = stats.wilcoxon(scores1, scores2)
                        
                        mean1 = np.mean(scores1)
                        mean2 = np.mean(scores2)
                        std1 = np.std(scores1)
                        std2 = np.std(scores2)
                        
                        print(f"\n{mode1} vs {mode2}:")
                        print(f"p-value: {p_value:.4f}")
                        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                        print(f"{mode1}: {mean1:.4f} ± {std1:.4f}")
                        print(f"{mode2}: {mean2:.4f} ± {std2:.4f}")
                        print("-" * 30)
    
    def run_feature_impact_experiment(self, mode='1_5'):
        """Experiment 3: Analyze impact of features on classification performance"""
        print("\nExperiment 3: Feature Impact Analysis")
        print("=" * 50)
        
        X, y = self.prepare_data(mode=mode)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.analyze_model_features(X_train, y_train, mode)
        
        self.analyze_review_length_impact(X, y, mode)
        
        self.analyze_bert_misclassifications(X_test, y_test, mode)
    
    def analyze_model_features(self, X_train, y_train, mode):
        """Analyze important features from TF-IDF and SVM coefficients"""
        print("\nAnalyzing Model Features:")
        print("=" * 50)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_train_tfidf_dense = X_train_tfidf.toarray()
        
        feature_names = vectorizer.get_feature_names_out()
        
        print("\nTop TF-IDF Features by Class:")
        print("-" * 30)
        
        class_tfidf = {}
        for class_label in sorted(set(y_train)):
            class_mask = (y_train == class_label)
            class_tfidf[class_label] = np.mean(X_train_tfidf_dense[class_mask], axis=0)
        
        for class_label, scores in class_tfidf.items():
            print(f"\nClass: {class_label}")
            top_indices = np.argsort(scores)[-20:]
            for idx in reversed(top_indices):
                print(f"{feature_names[idx]}: {scores[idx]:.4f}")
        
        print("\nTraining SVM to analyze coefficients...")
        svm_model = SVC(kernel='linear', C=1.0, probability=True)
        svm_model.fit(X_train_tfidf, y_train)
        
        print("\nTop SVM Coefficients by Class:")
        print("-" * 30)
        
        for i, class_label in enumerate(sorted(set(y_train))):
            print(f"\nClass: {class_label}")
            coef = svm_model.coef_[i].toarray().flatten()
            top_indices = np.argsort(coef)[-20:]
            for idx in reversed(top_indices):
                print(f"{feature_names[idx]}: {coef[idx]:.4f}")
    
    def analyze_review_length_impact(self, X, y, mode):
        print("\nAnalyzing Review Length Impact:")
        print("=" * 50)
        
        review_lengths = X.apply(len)
        
        length_categories = pd.qcut(review_lengths, q=4, labels=['Very Short', 'Short', 'Long', 'Very Long'])
        
        # Remove ROC AUC from metrics for this analysis
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        length_results = {model: {category: {metric: [] for metric in metrics} 
                                for category in length_categories.unique()} 
                         for model in self.models}
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for category in length_categories.unique():
            print(f"\nAnalyzing {category} reviews...")
            category_mask = (length_categories == category)
            X_category = X[category_mask]
            y_category = y[category_mask]
            
            if len(X_category) < 4:  # Skip if too few samples
                print(f"Skipping {category} category - too few samples")
                continue
                
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_category), 1):
                X_train, X_test = X_category.iloc[train_idx], X_category.iloc[test_idx]
                y_train, y_test = y_category.iloc[train_idx], y_category.iloc[test_idx]
                
                for model_name, model_class in [('SVM', self.svm), ('Naive Bayes', self.bayes)]:
                    try:
                        model, _, metrics_dict, _, _ = model_class.train_and_evaluate(
                            mode=mode, X_train=X_train, X_test=X_test, 
                            y_train=y_train, y_test=y_test)
                        
                        for metric in metrics:
                            if metric in metrics_dict:  # Only add if metric was successfully calculated
                                length_results[model_name][category][metric].append(metrics_dict[metric])
                    except Exception as e:
                        print(f"Error in {model_name} for {category}: {str(e)}")
                        continue
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_category, y_category, test_size=0.2, random_state=42, stratify=y_category)
                
                bert_model, _, bert_metrics, _, _ = self.bert.train_and_evaluate(
                    mode=mode, X_train=X_train, X_test=X_test, 
                    y_train=y_train, y_test=y_test)
                
                for metric in metrics:
                    if metric in bert_metrics:  # Only add if metric was successfully calculated
                        length_results['BERT'][category][metric] = [bert_metrics[metric]] * self.n_splits
            except Exception as e:
                print(f"Error in BERT for {category}: {str(e)}")
                continue
        
        print("\nReview Length Impact Analysis:")
        print("=" * 50)
        
        for model in self.models:
            print(f"\n{model} Performance by Review Length:")
            for metric in metrics:
                print(f"\n{metric.upper()}:")
                for category in length_categories.unique():
                    if length_results[model][category][metric]:
                        scores = length_results[model][category][metric]
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        print(f"{category}: {mean_score:.4f} ± {std_score:.4f}")
                    else:
                        print(f"{category}: No results available")
        
        self.plot_length_impact(length_results)
    
    def analyze_bert_misclassifications(self, X_test, y_test, mode):
        """Analyze BERT misclassifications to understand challenging cases"""
        print("\nAnalyzing BERT Misclassifications:")
        print("=" * 50)
        
        bert_model, tokenizer, metrics, y_pred, y_true = self.bert.train_and_evaluate(
            mode=mode, X_train=X_test, X_test=X_test, y_train=y_test, y_test=y_test)
        
        misclassified = X_test[y_pred != y_true]
        true_labels = y_true[y_pred != y_true]
        pred_labels = y_pred[y_pred != y_true]
        
        print(f"\nFound {len(misclassified)} misclassified examples")
        
        print("\nExample Misclassifications:")
        print("-" * 30)
        
        for i, (review, true_label, pred_label) in enumerate(zip(misclassified, true_labels, pred_labels)):
            if i >= 10:
                break
            
            print(f"\nExample {i+1}:")
            print(f"True Label: {true_label}")
            print(f"Predicted Label: {pred_label}")
            print(f"Review: {review[:200]}...")  # Show first 200 characters
            
            # Analyze review characteristics
            words = review.split()
            print(f"Review Length: {len(words)} words")
            print(f"Average Word Length: {np.mean([len(w) for w in words]):.2f}")
            print(f"Unique Words: {len(set(words))}")
            print("-" * 30)
    
    def plot_length_impact(self, length_results):
        """Create visualization for review length impact analysis"""
        plt.style.use('seaborn')
        
        for model in self.models:
            plt.figure(figsize=(12, 6))
            
            data = []
            for category in ['Very Short', 'Short', 'Long', 'Very Long']:
                for metric in self.metrics:
                    scores = length_results[model][category][metric]
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    data.append({
                        'Category': category,
                        'Metric': metric,
                        'Score': mean_score,
                        'Std': std_score
                    })
            
            df = pd.DataFrame(data)
            
            ax = sns.barplot(data=df, x='Category', y='Score', hue='Metric')
            
            for i, category in enumerate(df['Category'].unique()):
                for j, metric in enumerate(df['Metric'].unique()):
                    mask = (df['Category'] == category) & (df['Metric'] == metric)
                    score = df.loc[mask, 'Score'].iloc[0]
                    std = df.loc[mask, 'Std'].iloc[0]
                    ax.errorbar(i, score, yerr=std, fmt='none', color='black', capsize=5)
            
            plt.title(f'Review Length Impact on {model} Performance')
            plt.xlabel('Review Length Category')
            plt.ylabel('Score')
            plt.legend(title='Metric')
            plt.tight_layout()
            
            plt.show()
            plt.close()
    
    def run_all_experiments(self):
        all_results = self.run_model_comparison_experiment()
        self.run_mode_comparison_experiment(all_results)
        
        feature_results = self.run_feature_impact_experiment()
        
        return all_results, feature_results

if __name__ == "__main__":
    comparison = ModelComparison()
    all_results, feature_results = comparison.run_all_experiments()
