"""
Problem 5: A/B Test Statistical Analysis

Build a service to analyze A/B test results and determine statistical significance.

Shopify Context:
- Test new features (checkout flow, recommendation algorithms, UI changes)
- Decide whether to launch or rollback
- Ensure statistical rigor

Interview Focus:
- Statistical concepts (p-values, confidence intervals, power)
- Handling edge cases (small samples, early stopping)
- Clean API design
- Production considerations
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from enum import Enum


class TestType(Enum):
    """Type of A/B test."""
    CONVERSION_RATE = "conversion_rate"  # Binary outcome (converted or not)
    CONTINUOUS = "continuous"  # Continuous metric (revenue, time on site)
    COUNT = "count"  # Count data (# of purchases, # of clicks)


class Decision(Enum):
    """A/B test decision."""
    LAUNCH = "launch"  # Variant B is better
    ROLLBACK = "rollback"  # Variant A is better
    INCONCLUSIVE = "inconclusive"  # No significant difference


@dataclass
class TestResult:
    """
    Results of A/B test analysis.

    Interview Tip: "Using dataclass for clean data representation.
    Makes the API self-documenting."
    """
    variant_a_metric: float
    variant_b_metric: float
    lift: float  # Percentage improvement
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    decision: Decision
    sample_size_a: int
    sample_size_b: int
    power: Optional[float] = None  # Statistical power
    min_detectable_effect: Optional[float] = None


class ABTestAnalyzer:
    """
    Analyze A/B test results with statistical rigor.

    Interview Tip: "This implements standard statistical tests used in
    industry. The key is choosing the right test for the data type and
    handling edge cases properly."
    """

    def __init__(self, alpha: float = 0.05, power_target: float = 0.8):
        """
        Initialize analyzer.

        Args:
            alpha: Significance level (default 0.05 = 95% confidence)
            power_target: Desired statistical power (default 0.8)

        Interview Tip: "Alpha of 0.05 is standard in industry.
        Power of 0.8 means 80% chance of detecting a real effect."
        """
        self.alpha = alpha
        self.power_target = power_target

    def analyze_conversion_rate(self,
                               conversions_a: int,
                               visitors_a: int,
                               conversions_b: int,
                               visitors_b: int) -> TestResult:
        """
        Analyze conversion rate test using two-proportion z-test.

        Args:
            conversions_a: Number of conversions in variant A (control)
            visitors_a: Total visitors in variant A
            conversions_b: Number of conversions in variant B (treatment)
            visitors_b: Total visitors in variant B

        Returns:
            TestResult with analysis

        Interview Tip: "For conversion rates, we use a two-proportion z-test.
        This is appropriate for binary outcomes like purchased/didn't purchase."
        """
        # Input validation
        if visitors_a <= 0 or visitors_b <= 0:
            raise ValueError("Sample sizes must be positive")
        if conversions_a < 0 or conversions_b < 0:
            raise ValueError("Conversions cannot be negative")
        if conversions_a > visitors_a or conversions_b > visitors_b:
            raise ValueError("Conversions cannot exceed visitors")

        # Calculate conversion rates
        rate_a = conversions_a / visitors_a
        rate_b = conversions_b / visitors_b

        # Calculate pooled proportion for z-test
        pooled_conv = conversions_a + conversions_b
        pooled_visitors = visitors_a + visitors_b
        pooled_rate = pooled_conv / pooled_visitors

        # Standard error
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/visitors_a + 1/visitors_b))

        # Z-statistic
        if se == 0:
            # No variance, can't do test
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (rate_b - rate_a) / se

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval for difference in proportions
        se_diff = np.sqrt(rate_a * (1 - rate_a) / visitors_a +
                         rate_b * (1 - rate_b) / visitors_b)

        ci_lower = (rate_b - rate_a) - 1.96 * se_diff
        ci_upper = (rate_b - rate_a) + 1.96 * se_diff

        # Lift (percentage change)
        lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0

        # Decision
        is_significant = p_value < self.alpha

        if is_significant:
            if rate_b > rate_a:
                decision = Decision.LAUNCH
            else:
                decision = Decision.ROLLBACK
        else:
            decision = Decision.INCONCLUSIVE

        # Calculate statistical power
        effect_size = abs(rate_b - rate_a)
        power = self._calculate_power_proportion(
            visitors_a, visitors_b, rate_a, effect_size
        )

        # Minimum detectable effect at target power
        mde = self._calculate_mde_proportion(visitors_a, visitors_b, rate_a)

        return TestResult(
            variant_a_metric=rate_a,
            variant_b_metric=rate_b,
            lift=lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            decision=decision,
            sample_size_a=visitors_a,
            sample_size_b=visitors_b,
            power=power,
            min_detectable_effect=mde
        )

    def analyze_continuous_metric(self,
                                 values_a: List[float],
                                 values_b: List[float]) -> TestResult:
        """
        Analyze continuous metric (e.g., revenue, time on site) using t-test.

        Args:
            values_a: Values from variant A
            values_b: Values from variant B

        Returns:
            TestResult with analysis

        Interview Tip: "For continuous metrics like revenue, we use Welch's
        t-test which doesn't assume equal variances. More robust than
        Student's t-test."
        """
        # Input validation
        if len(values_a) == 0 or len(values_b) == 0:
            raise ValueError("Need at least one observation per variant")

        values_a = np.array(values_a)
        values_b = np.array(values_b)

        # Calculate means
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

        # Confidence interval for difference in means
        # Using Welch-Satterthwaite degrees of freedom
        var_a = np.var(values_a, ddof=1)
        var_b = np.var(values_b, ddof=1)
        n_a = len(values_a)
        n_b = len(values_b)

        se_diff = np.sqrt(var_a/n_a + var_b/n_b)

        # Degrees of freedom (Welch-Satterthwaite)
        df = (var_a/n_a + var_b/n_b)**2 / (
            (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
        )

        t_critical = stats.t.ppf(1 - self.alpha/2, df)

        ci_lower = (mean_b - mean_a) - t_critical * se_diff
        ci_upper = (mean_b - mean_a) + t_critical * se_diff

        # Lift
        lift = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0

        # Decision
        is_significant = p_value < self.alpha

        if is_significant:
            if mean_b > mean_a:
                decision = Decision.LAUNCH
            else:
                decision = Decision.ROLLBACK
        else:
            decision = Decision.INCONCLUSIVE

        # Calculate power (Cohen's d effect size)
        pooled_std = np.sqrt((var_a + var_b) / 2)
        if pooled_std > 0:
            cohens_d = abs(mean_b - mean_a) / pooled_std
            power = self._calculate_power_ttest(n_a, n_b, cohens_d)
        else:
            power = None

        return TestResult(
            variant_a_metric=mean_a,
            variant_b_metric=mean_b,
            lift=lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            decision=decision,
            sample_size_a=n_a,
            sample_size_b=n_b,
            power=power
        )

    def check_sample_size(self,
                         baseline_rate: float,
                         mde: float,
                         power: float = 0.8,
                         alpha: float = 0.05) -> int:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Expected baseline conversion rate
            mde: Minimum detectable effect (as proportion, e.g., 0.02 for 2%)
            power: Desired statistical power (default 0.8)
            alpha: Significance level (default 0.05)

        Returns:
            Required sample size per variant

        Interview Tip: "This helps answer the question: 'How long should
        we run the test?' Critical for experiment planning."
        """
        # For two-proportion test
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate + mde

        # Pooled proportion under null
        p_pooled = (p1 + p2) / 2

        # Sample size formula
        n = (
            (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 /
            (p2 - p1)**2
        )

        return int(np.ceil(n))

    def early_stopping_check(self,
                           conversions_a: int,
                           visitors_a: int,
                           conversions_b: int,
                           visitors_b: int,
                           max_sample_size: int) -> Dict:
        """
        Check if test can be stopped early (with corrections for peeking).

        Args:
            conversions_a: Current conversions in A
            visitors_a: Current visitors in A
            conversions_b: Current conversions in B
            visitors_b: Current visitors in B
            max_sample_size: Planned maximum sample size per variant

        Returns:
            Dict with stopping recommendation and reasoning

        Interview Tip: "Early stopping seems attractive but requires
        careful handling to avoid false positives. Need to adjust
        significance levels (alpha spending)."
        """
        # Calculate progress
        progress = max(visitors_a, visitors_b) / max_sample_size

        # Sequential testing: adjust alpha using Pocock boundary
        # (conservative approach)
        num_looks = 5  # Assume we'll peek 5 times
        adjusted_alpha = self.alpha / num_looks

        # Run test with adjusted alpha
        original_alpha = self.alpha
        self.alpha = adjusted_alpha
        result = self.analyze_conversion_rate(
            conversions_a, visitors_a, conversions_b, visitors_b
        )
        self.alpha = original_alpha

        recommendation = {
            'can_stop': False,
            'decision': Decision.INCONCLUSIVE,
            'progress': progress,
            'adjusted_alpha': adjusted_alpha,
            'p_value': result.p_value,
            'reason': ''
        }

        if progress < 0.5:
            recommendation['reason'] = "Sample size too small (< 50% of target)"
        elif result.is_significant:
            recommendation['can_stop'] = True
            recommendation['decision'] = result.decision
            recommendation['reason'] = f"Significant result with adjusted alpha={adjusted_alpha:.4f}"
        elif progress >= 1.0:
            recommendation['can_stop'] = True
            recommendation['decision'] = Decision.INCONCLUSIVE
            recommendation['reason'] = "Reached maximum sample size without significance"
        else:
            recommendation['reason'] = f"Not yet significant. Continue to {max_sample_size} samples."

        return recommendation

    def _calculate_power_proportion(self, n_a: int, n_b: int,
                                   p_a: float, effect: float) -> float:
        """Calculate statistical power for proportion test."""
        p_b = p_a + effect
        pooled_p = (p_a + p_b) / 2

        # Standard error under alternative hypothesis
        se_alt = np.sqrt(p_a * (1-p_a) / n_a + p_b * (1-p_b) / n_b)

        # Critical value
        z_alpha = stats.norm.ppf(1 - self.alpha/2)

        # Standard error under null
        se_null = np.sqrt(pooled_p * (1-pooled_p) * (1/n_a + 1/n_b))

        # Non-centrality parameter
        delta = effect / se_alt

        # Power
        power = 1 - stats.norm.cdf(z_alpha - delta)

        return power

    def _calculate_power_ttest(self, n_a: int, n_b: int,
                              cohens_d: float) -> float:
        """Calculate statistical power for t-test."""
        # Using approximation for Welch's t-test
        df = n_a + n_b - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)

        # Non-centrality parameter
        ncp = cohens_d * np.sqrt(n_a * n_b / (n_a + n_b))

        # Power
        power = 1 - stats.nct.cdf(t_critical, df, ncp)

        return power

    def _calculate_mde_proportion(self, n_a: int, n_b: int,
                                 p_a: float) -> float:
        """Calculate minimum detectable effect at target power."""
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(self.power_target)

        # Solve for mde
        # Approximate solution
        se_pooled = np.sqrt(2 * p_a * (1 - p_a) / n_a)  # Assuming equal n
        mde = (z_alpha + z_beta) * se_pooled

        return mde


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("A/B TEST ANALYZER DEMO")
    print("="*60 + "\n")

    analyzer = ABTestAnalyzer(alpha=0.05)

    # Example 1: Conversion rate test
    print("1. CONVERSION RATE TEST")
    print("-" * 40)
    print("Scenario: Testing new checkout flow")
    print("Control (A): 150 conversions / 10000 visitors = 1.5%")
    print("Treatment (B): 180 conversions / 10000 visitors = 1.8%\n")

    result = analyzer.analyze_conversion_rate(
        conversions_a=150,
        visitors_a=10000,
        conversions_b=180,
        visitors_b=10000
    )

    print(f"Results:")
    print(f"  Variant A rate: {result.variant_a_metric:.2%}")
    print(f"  Variant B rate: {result.variant_b_metric:.2%}")
    print(f"  Lift: {result.lift:+.1f}%")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    print(f"  Significant: {result.is_significant}")
    print(f"  Decision: {result.decision.value}")
    print(f"  Statistical power: {result.power:.2%}")
    print(f"  Min detectable effect: {result.min_detectable_effect:.2%}\n")

    # Example 2: Revenue test (continuous)
    print("\n2. CONTINUOUS METRIC TEST (Revenue)")
    print("-" * 40)

    np.random.seed(42)
    revenue_a = np.random.gamma(shape=2, scale=50, size=1000)  # Mean ~$100
    revenue_b = np.random.gamma(shape=2, scale=55, size=1000)  # Mean ~$110

    print(f"Control (A): {len(revenue_a)} users, mean revenue ${np.mean(revenue_a):.2f}")
    print(f"Treatment (B): {len(revenue_b)} users, mean revenue ${np.mean(revenue_b):.2f}\n")

    result = analyzer.analyze_continuous_metric(
        values_a=revenue_a.tolist(),
        values_b=revenue_b.tolist()
    )

    print(f"Results:")
    print(f"  Variant A mean: ${result.variant_a_metric:.2f}")
    print(f"  Variant B mean: ${result.variant_b_metric:.2f}")
    print(f"  Lift: {result.lift:+.1f}%")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  95% CI: [${result.confidence_interval[0]:.2f}, ${result.confidence_interval[1]:.2f}]")
    print(f"  Significant: {result.is_significant}")
    print(f"  Decision: {result.decision.value}\n")

    # Example 3: Sample size calculation
    print("\n3. SAMPLE SIZE CALCULATOR")
    print("-" * 40)

    baseline = 0.02  # 2% baseline conversion rate
    mde = 0.002  # Want to detect 0.2% absolute change (10% relative)

    required_n = analyzer.check_sample_size(
        baseline_rate=baseline,
        mde=mde,
        power=0.8,
        alpha=0.05
    )

    print(f"Baseline conversion rate: {baseline:.1%}")
    print(f"Minimum detectable effect: {mde:.1%} ({mde/baseline:.0%} relative)")
    print(f"Required sample size per variant: {required_n:,}")
    print(f"Total required: {required_n * 2:,}\n")

    # Example 4: Early stopping
    print("\n4. EARLY STOPPING CHECK")
    print("-" * 40)

    max_n = 10000
    current_conv_a = 80
    current_n_a = 5000
    current_conv_b = 110
    current_n_b = 5000

    early_stop = analyzer.early_stopping_check(
        conversions_a=current_conv_a,
        visitors_a=current_n_a,
        conversions_b=current_conv_b,
        visitors_b=current_n_b,
        max_sample_size=max_n
    )

    print(f"Current progress: {early_stop['progress']:.0%} of target")
    print(f"P-value: {early_stop['p_value']:.4f}")
    print(f"Adjusted alpha: {early_stop['adjusted_alpha']:.4f}")
    print(f"Can stop: {early_stop['can_stop']}")
    print(f"Recommendation: {early_stop['reason']}\n")

    print("\n" + "="*60)
    print("INTERVIEW DISCUSSION POINTS")
    print("="*60)
    print("""
    1. STATISTICAL TESTS:
       - Conversion rate: Two-proportion z-test
       - Continuous metrics: Welch's t-test (robust to unequal variances)
       - Count data: Poisson test or negative binomial

    2. MULTIPLE TESTING:
       - Testing multiple metrics increases false positive rate
       - Use Bonferroni correction or FDR control
       - Primary metric vs secondary metrics

    3. EARLY STOPPING:
       - Peeking at results inflates Type I error
       - Need alpha spending functions (Pocock, O'Brien-Fleming)
       - Trade-off: Stop early vs statistical rigor

    4. SAMPLE SIZE:
       - Depends on baseline rate, MDE, power, alpha
       - Smaller MDE = larger sample needed
       - Higher power = larger sample needed

    5. PRACTICAL CONSIDERATIONS:
       - Novelty effects (users try new things)
       - Seasonal effects (don't test over Black Friday)
       - Selection bias (who opts into test?)
       - Long-term vs short-term effects

    6. PRODUCTION:
       - Automated experiment analysis
       - A/A tests to verify system
       - Traffic allocation (50/50, multi-armed bandits)
       - Segment analysis (mobile vs desktop)
    """)
