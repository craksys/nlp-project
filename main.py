import numpy as np
from scipy import stats
import pandas as pd
from svm.svm import SVMClassifier
from bayes.bayes import NaiveBayesClassifier
from bert.bert import BERTClassifier
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk import word_tokenize

class ModelComparison:
    def __init__(self, data_path="tripadvisor_hotel_reviews.csv", n_splits=5):
        self.svm = SVMClassifier(data_path)
        self.bayes = NaiveBayesClassifier(data_path)
        self.bert = BERTClassifier(data_path)
        self.n_splits = n_splits
        self.modes = ['1_5', 'with_neutral', 'without_neutral']
        self.models = ['SVM', 'Naive Bayes', 'BERT']
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Define feature combinations for experiment 3
        self.feature_combinations = {
            'baseline': {
                'lowercase': True,
                'remove_special_chars': True,
                'remove_stopwords': True,
                'lemmatize': True
            },
            'no_lemmatization': {
                'lowercase': True,
                'remove_special_chars': True,
                'remove_stopwords': True,
                'lemmatize': False
            },
            'keep_stopwords': {
                'lowercase': True,
                'remove_special_chars': True,
                'remove_stopwords': False,
                'lemmatize': True
            },
            'keep_special_chars': {
                'lowercase': True,
                'remove_special_chars': False,
                'remove_stopwords': True,
                'lemmatize': True
            },
            'no_lowercase': {
                'lowercase': False,
                'remove_special_chars': True,
                'remove_stopwords': True,
                'lemmatize': True
            }
        }
    
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
        
        # Initialize results storage
        cv_results = {model: {metric: [] for metric in self.metrics} for model in self.models}
        
        # Prepare data once
        X, y = self.prepare_data(mode=mode)
        
        # Initialize KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        
        # For BERT, use a single train/test split
        X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        # Train and evaluate BERT once
        print("\nTraining BERT model (single split)...")
        bert_model, bert_tokenizer, bert_metrics, _, _ = self.bert.train_and_evaluate(
            mode=mode, X_train=X_train_bert, X_test=X_test_bert, 
            y_train=y_train_bert, y_test=y_test_bert)
        for metric in self.metrics:
            cv_results['BERT'][metric] = [bert_metrics[metric]] * self.n_splits
        
        # Run cross validation for SVM and Naive Bayes
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{self.n_splits}")
            print("-" * 30)
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train and evaluate SVM
            svm_model, svm_vectorizer, svm_metrics, _, _ = self.svm.train_and_evaluate(
                mode=mode, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            for metric in self.metrics:
                cv_results['SVM'][metric].append(svm_metrics[metric])
            
            # Train and evaluate Naive Bayes
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
                        # Get performance scores for both models
                        scores1 = cv_results[model1][metric]
                        scores2 = cv_results[model2][metric]
                        
                        # Perform Wilcoxon test
                        statistic, p_value = stats.wilcoxon(scores1, scores2)
                        
                        # Determine significance and direction
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
                # Perform Friedman test (non-parametric alternative to repeated measures ANOVA)
                data = [all_results[mode][model][metric] for mode in self.modes]
                statistic, p_value = stats.friedmanchisquare(*data)
                
                print(f"Friedman test p-value: {p_value:.4f}")
                print("Significant difference between modes:", "Yes" if p_value < 0.05 else "No")
                
                # Print mean values for each mode
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
            
            # Prepare data for plotting
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
            
            # Create plot
            sns.lineplot(data=df, x='Mode', y='Value', hue='Metric', marker='o')
            plt.title(f'Metrics Across Modes - {model}')
            plt.xlabel('Classification Mode')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'{model.lower()}_metrics_by_mode.png')
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
            # Print header
            print("Model".ljust(15), end="")
            for model in self.models:
                print(f"{model}".ljust(15), end="")
            print("\n" + "-" * 60)
            
            # Print rows
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
            
            # Run cross validation
            cv_results = self.run_cross_validation(mode=mode)
            all_results[mode] = cv_results
            
            # Perform Wilcoxon tests
            significance_matrix = self.perform_wilcoxon_tests(cv_results)
            
            # Print significance matrix
            self.print_significance_matrix(significance_matrix, mode)
            
            # Print summary statistics
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
                
                # Compare each pair of modes
                for i, mode1 in enumerate(self.modes):
                    for mode2 in self.modes[i+1:]:
                        # Get performance scores for both modes
                        scores1 = all_results[mode1][model][metric]
                        scores2 = all_results[mode2][model][metric]
                        
                        # Perform Wilcoxon test
                        statistic, p_value = stats.wilcoxon(scores1, scores2)
                        
                        # Calculate mean values
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
        """Experiment 3: Analyze impact of different textual features"""
        print("\nExperiment 3: Feature Impact Analysis")
        print("=" * 50)
        
        feature_results = {}
        
        # Prepare base data
        X, y = self.prepare_data(mode=mode)
        
        # Test each feature combination
        for feature_name, features in self.feature_combinations.items():
            print(f"\nTesting feature combination: {feature_name}")
            print("-" * 30)
            
            # Apply feature combination
            X_processed = X.apply(lambda x: self.preprocess_text_with_features(x, features))
            
            # Initialize KFold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            # Store results for this feature combination
            feature_results[feature_name] = {model: {metric: [] for metric in self.metrics} 
                                           for model in self.models}
            
            # Run cross validation
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_processed), 1):
                print(f"\nFold {fold}/{self.n_splits}")
                
                X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train and evaluate each model
                for model_name, model_class in [('SVM', self.svm), ('Naive Bayes', self.bayes)]:
                    model, _, metrics, _, _ = model_class.train_and_evaluate(
                        mode=mode, X_train=X_train, X_test=X_test, 
                        y_train=y_train, y_test=y_test)
                    
                    for metric in self.metrics:
                        feature_results[feature_name][model_name][metric].append(metrics[metric])
            
            # For BERT, use single train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y)
            
            bert_model, _, bert_metrics, _, _ = self.bert.train_and_evaluate(
                mode=mode, X_train=X_train, X_test=X_test, 
                y_train=y_train, y_test=y_test)
            
            for metric in self.metrics:
                feature_results[feature_name]['BERT'][metric] = [bert_metrics[metric]] * self.n_splits
        
        # Analyze results
        self.analyze_feature_impact(feature_results)
        
        # Create visualizations
        self.plot_feature_impact(feature_results)
        
        return feature_results
    
    def analyze_feature_impact(self, feature_results):
        """Analyze the impact of different features on model performance"""
        print("\nFeature Impact Analysis Results:")
        print("=" * 50)
        
        for model in self.models:
            print(f"\n{model} Feature Impact:")
            for metric in self.metrics:
                print(f"\n{metric.upper()}:")
                
                # Get baseline performance
                baseline_scores = feature_results['baseline'][model][metric]
                baseline_mean = np.mean(baseline_scores)
                
                # Compare each feature combination to baseline
                for feature_name, results in feature_results.items():
                    if feature_name != 'baseline':
                        scores = results[model][metric]
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        
                        # Calculate percentage change from baseline
                        pct_change = ((mean_score - baseline_mean) / baseline_mean) * 100
                        
                        print(f"\n{feature_name}:")
                        print(f"Mean: {mean_score:.4f} ± {std_score:.4f}")
                        print(f"Change from baseline: {pct_change:+.2f}%")
                        
                        # Perform statistical test
                        statistic, p_value = stats.wilcoxon(baseline_scores, scores)
                        print(f"p-value: {p_value:.4f}")
                        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                        print("-" * 30)
    
    def plot_feature_impact(self, feature_results):
        """Create visualizations for feature impact analysis"""
        plt.style.use('seaborn')
        
        for model in self.models:
            plt.figure(figsize=(15, 8))
            
            # Prepare data for plotting
            data = []
            for feature_name, results in feature_results.items():
                for metric in self.metrics:
                    scores = results[model][metric]
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    data.append({
                        'Feature': feature_name,
                        'Metric': metric,
                        'Score': mean_score,
                        'Std': std_score
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar plot
            ax = sns.barplot(data=df, x='Feature', y='Score', hue='Metric')
            
            # Add error bars
            for i, feature in enumerate(df['Feature'].unique()):
                for j, metric in enumerate(df['Metric'].unique()):
                    mask = (df['Feature'] == feature) & (df['Metric'] == metric)
                    score = df.loc[mask, 'Score'].iloc[0]
                    std = df.loc[mask, 'Std'].iloc[0]
                    ax.errorbar(i, score, yerr=std, fmt='none', color='black', capsize=5)
            
            plt.title(f'Feature Impact on {model} Performance')
            plt.xlabel('Feature Combination')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'{model.lower()}_feature_impact.png')
            plt.close()
    
    def run_all_experiments(self):
        """Run all experiments and create visualizations"""
        # Run Experiment 1: Model Comparison
        all_results = self.run_model_comparison_experiment()
        
        # Run Experiment 2: Mode Comparison
        self.run_mode_comparison_experiment(all_results)
        
        # Run Experiment 3: Feature Impact Analysis
        feature_results = self.run_feature_impact_experiment()
        
        return all_results, feature_results

if __name__ == "__main__":
    comparison = ModelComparison()
    all_results, feature_results = comparison.run_all_experiments()
