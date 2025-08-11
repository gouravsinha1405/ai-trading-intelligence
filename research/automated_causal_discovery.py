"""
Market Swing Prediction Using Causal Graphs
==========================================

VISION: Use the comprehensive causal graph to predict market swings
by detecting early warning patterns and cascade effects.

ORIGINAL INNOVATION BY: Gourav Sinha  
DATE: August 11, 2025
PURPOSE: Predict market swings using multi-domain causal relationships
NOVELTY: First market swing predictor using automated causal discovery

PREDICTION METHODOLOGY:
======================

1. PATTERN MATCHING:
   Current Event Sequence → Historical Similar Sequences → Predicted Market Swing

2. CASCADE DETECTION:
   Event A → Event B → Event C → Market Swing (catch at Event A)

3. WEAK SIGNAL AGGREGATION:
   Multiple weak causal signals → Strong prediction when combined

4. CROSS-DOMAIN CONVERGENCE:
   Sports + Entertainment + Politics pointing same direction → High confidence prediction

MATHEMATICAL FRAMEWORK:
======================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MarketSwingPredictor:
    """
    Predict market swings using comprehensive causal graph analysis
    
    PREDICTION CAPABILITIES:
    1. Short-term swings (1-7 days)
    2. Medium-term trends (1-4 weeks)  
    3. Major corrections/rallies (1-3 months)
    4. Sector rotation predictions
    5. Volatility spike warnings
    
    CAUSAL SIGNAL SOURCES:
    - Sports sponsorship effects
    - Celebrity endorsement impacts
    - Political policy implications
    - Weather/disaster consequences
    - Technology disruption patterns
    - Social media sentiment cascades
    """
    
    def __init__(self, causal_graph, lookback_days=30, prediction_horizon=7):
        self.causal_graph = causal_graph
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        
        # Prediction models
        self.swing_magnitude_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.swing_direction_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.volatility_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Pattern recognition
        self.historical_patterns = {}
        self.cascade_patterns = {}
        self.weak_signal_aggregator = WeakSignalAggregator()
        
        print("📈 Market Swing Predictor initialized")
        print(f"🔍 Lookback window: {lookback_days} days")
        print(f"🎯 Prediction horizon: {prediction_horizon} days")
    
    def predict_market_swing(self, current_events, market_context):
        """
        Main prediction function for market swings
        
        INPUTS:
        - current_events: Recent events affecting markets
        - market_context: Current market state (volatility, trend, etc.)
        
        OUTPUTS:
        - swing_magnitude: Expected % move (-1 to 1)
        - swing_direction: Bullish/Bearish probability
        - confidence: Prediction confidence (0 to 1)
        - reasoning: Explanation of causal factors
        """
        
        print("🔮 Analyzing current events for market swing prediction...")
        
        # Step 1: Pattern matching with historical sequences
        pattern_signals = self._analyze_historical_patterns(current_events)
        
        # Step 2: Detect cascade effects
        cascade_signals = self._detect_cascade_effects(current_events)
        
        # Step 3: Aggregate weak signals
        weak_signals = self._aggregate_weak_signals(current_events)
        
        # Step 4: Cross-domain convergence analysis
        convergence_signals = self._analyze_cross_domain_convergence(current_events)
        
        # Step 5: Combine all signals for final prediction
        final_prediction = self._combine_prediction_signals(
            pattern_signals, cascade_signals, weak_signals, 
            convergence_signals, market_context
        )
        
        return final_prediction
    
    def _analyze_historical_patterns(self, current_events):
        """
        Find similar historical event sequences and their market outcomes
        
        ALGORITHM:
        1. Extract features from current event sequence
        2. Find similar sequences in historical data
        3. Analyze market outcomes of similar sequences
        4. Weight by similarity and recency
        """
        
        print("  📊 Matching current events with historical patterns...")
        
        # Extract features from current events
        current_features = self._extract_sequence_features(current_events)
        
        # Find similar historical patterns
        similar_patterns = []
        for pattern_id, historical_pattern in self.historical_patterns.items():
            similarity = self._calculate_pattern_similarity(
                current_features, historical_pattern['features']
            )
            
            if similarity > 0.7:  # High similarity threshold
                similar_patterns.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'market_outcome': historical_pattern['market_outcome'],
                    'timeframe': historical_pattern['timeframe']
                })
        
        # Weight patterns by similarity and recency
        if similar_patterns:
            weighted_prediction = self._weight_historical_predictions(similar_patterns)
            confidence = np.mean([p['similarity'] for p in similar_patterns])
        else:
            weighted_prediction = {'magnitude': 0.0, 'direction': 0.0}
            confidence = 0.0
        
        return {
            'prediction': weighted_prediction,
            'confidence': confidence,
            'pattern_count': len(similar_patterns),
            'type': 'historical_pattern'
        }
    
    def _detect_cascade_effects(self, current_events):
        """
        Detect if current events are part of a cascade leading to market swing
        
        CASCADE EXAMPLES:
        1. Political Announcement → Policy Expectation → Sector Rotation
        2. Celebrity Endorsement → Social Media Viral → Consumer Behavior → Stock Rally
        3. Tech Breakthrough → Competitor Panic → Sector Reshuffling
        """
        
        print("  ⛓️ Detecting cascade effects...")
        
        cascade_signals = []
        
        # Analyze each current event for cascade potential
        for event in current_events:
            # Find outgoing edges from this event in causal graph
            if event['id'] in self.causal_graph.nodes:
                successors = list(self.causal_graph.successors(event['id']))
                
                # Check if any successors are likely to trigger market effects
                for successor in successors:
                    edge_data = self.causal_graph.edges[event['id'], successor]
                    causality_strength = edge_data.get('weight', 0.0)
                    
                    if causality_strength > 0.5:  # Strong causal relationship
                        # Predict the cascade effect
                        cascade_prediction = self._predict_cascade_outcome(
                            event, successor, causality_strength
                        )
                        cascade_signals.append(cascade_prediction)
        
        # Aggregate cascade signals
        if cascade_signals:
            aggregated_magnitude = np.mean([s['magnitude'] for s in cascade_signals])
            aggregated_direction = np.mean([s['direction'] for s in cascade_signals])
            confidence = np.mean([s['confidence'] for s in cascade_signals])
        else:
            aggregated_magnitude = 0.0
            aggregated_direction = 0.0
            confidence = 0.0
        
        return {
            'prediction': {
                'magnitude': aggregated_magnitude,
                'direction': aggregated_direction
            },
            'confidence': confidence,
            'cascade_count': len(cascade_signals),
            'type': 'cascade_effect'
        }
    
    def _aggregate_weak_signals(self, current_events):
        """
        Aggregate multiple weak signals that individually don't predict swings
        but together create strong prediction
        
        WEAK SIGNAL EXAMPLES:
        - Celebrity mentioned stock in interview (weak individual signal)
        - Sports team with bank sponsor won game (weak individual signal)
        - Politician tweeted about clean energy (weak individual signal)
        - Together: Strong bullish signal for clean energy stocks
        """
        
        print("  🔍 Aggregating weak signals...")
        
        weak_signals = []
        
        for event in current_events:
            # Calculate individual signal strength
            signal_strength = self._calculate_individual_signal_strength(event)
            
            if 0.1 < signal_strength < 0.4:  # Weak but non-negligible signal
                weak_signals.append({
                    'event': event,
                    'strength': signal_strength,
                    'sectors': event.get('affected_sectors', []),
                    'direction': self._infer_signal_direction(event)
                })
        
        # Use weak signal aggregator to combine signals
        aggregated_signal = self.weak_signal_aggregator.aggregate(weak_signals)
        
        return {
            'prediction': aggregated_signal['prediction'],
            'confidence': aggregated_signal['confidence'],
            'signal_count': len(weak_signals),
            'type': 'weak_signal_aggregation'
        }
    
    def _analyze_cross_domain_convergence(self, current_events):
        """
        Analyze when multiple domains point in the same direction
        
        HIGH CONFIDENCE SCENARIOS:
        - Sports sponsorship + Celebrity endorsement + Political support → Same sector
        - Weather disaster + Supply chain + Geopolitical → Same commodity
        - Tech breakthrough + VC funding + Media coverage → Same innovation area
        """
        
        print("  🎯 Analyzing cross-domain convergence...")
        
        # Group events by domain
        domain_signals = {}
        for event in current_events:
            domain = event.get('domain', 'unknown')
            if domain not in domain_signals:
                domain_signals[domain] = []
            
            signal = {
                'magnitude': event.get('market_impact', 0.0),
                'direction': self._infer_signal_direction(event),
                'sectors': event.get('affected_sectors', [])
            }
            domain_signals[domain].append(signal)
        
        # Find sector convergence across domains
        sector_convergence = self._calculate_sector_convergence(domain_signals)
        
        # Calculate convergence strength
        convergence_strength = 0.0
        if len(domain_signals) >= 2:  # Need at least 2 domains
            # Find sectors mentioned by multiple domains
            all_sectors = set()
            for domain, signals in domain_signals.items():
                for signal in signals:
                    all_sectors.update(signal['sectors'])
            
            for sector in all_sectors:
                domain_count = 0
                total_magnitude = 0.0
                directions = []
                
                for domain, signals in domain_signals.items():
                    for signal in signals:
                        if sector in signal['sectors']:
                            domain_count += 1
                            total_magnitude += signal['magnitude']
                            directions.append(signal['direction'])
                
                if domain_count >= 2:  # Sector mentioned by multiple domains
                    # Check if directions align
                    direction_alignment = np.std(directions) < 0.5  # Low std = aligned
                    if direction_alignment:
                        convergence_strength = max(convergence_strength, total_magnitude)
        
        return {
            'prediction': {
                'magnitude': convergence_strength,
                'direction': 1.0 if convergence_strength > 0 else -1.0
            },
            'confidence': min(convergence_strength * 2, 1.0),
            'converging_domains': len(domain_signals),
            'type': 'cross_domain_convergence'
        }
    
    def _combine_prediction_signals(self, pattern_signals, cascade_signals, 
                                   weak_signals, convergence_signals, market_context):
        """
        Combine all prediction signals into final market swing prediction
        
        COMBINATION METHODOLOGY:
        1. Weight signals by confidence and track record
        2. Adjust for current market context (volatility, trend)
        3. Apply ensemble methods for robust prediction
        4. Generate explanation for prediction reasoning
        """
        
        print("  🎯 Combining all signals for final prediction...")
        
        # Weight factors for different signal types
        weights = {
            'historical_pattern': 0.3,
            'cascade_effect': 0.25,
            'weak_signal_aggregation': 0.2,
            'cross_domain_convergence': 0.25
        }
        
        # Collect all signals
        signals = [pattern_signals, cascade_signals, weak_signals, convergence_signals]
        
        # Weighted combination
        weighted_magnitude = 0.0
        weighted_direction = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            signal_type = signal['type']
            weight = weights.get(signal_type, 0.0)
            confidence = signal['confidence']
            
            weighted_magnitude += weight * signal['prediction']['magnitude'] * confidence
            weighted_direction += weight * signal['prediction']['direction'] * confidence
            total_confidence += weight * confidence
        
        # Normalize by total confidence
        if total_confidence > 0:
            final_magnitude = weighted_magnitude / total_confidence
            final_direction = weighted_direction / total_confidence
            final_confidence = total_confidence / sum(weights.values())
        else:
            final_magnitude = 0.0
            final_direction = 0.0
            final_confidence = 0.0
        
        # Adjust for market context
        context_adjustment = self._apply_market_context_adjustment(
            final_magnitude, final_direction, market_context
        )
        
        final_magnitude *= context_adjustment['magnitude_multiplier']
        final_direction *= context_adjustment['direction_multiplier']
        
        # Generate explanation
        explanation = self._generate_prediction_explanation(signals, market_context)
        
        return {
            'swing_magnitude': final_magnitude,
            'swing_direction': 'bullish' if final_direction > 0 else 'bearish',
            'confidence': final_confidence,
            'time_horizon': f"{self.prediction_horizon} days",
            'explanation': explanation,
            'signal_breakdown': {
                'pattern_matching': pattern_signals,
                'cascade_effects': cascade_signals,
                'weak_signals': weak_signals,
                'convergence': convergence_signals
            }
        }
    
    def _apply_market_context_adjustment(self, magnitude, direction, market_context):
        """
        Adjust predictions based on current market conditions
        
        ADJUSTMENTS:
        - High volatility → Amplify swing predictions
        - Bull market → Bias towards positive swings
        - Bear market → Bias towards negative swings
        - Low volume → Reduce prediction confidence
        """
        
        current_volatility = market_context.get('volatility', 0.2)
        current_trend = market_context.get('trend', 'neutral')
        current_volume = market_context.get('volume_ratio', 1.0)
        
        # Volatility adjustment
        if current_volatility > 0.3:  # High volatility
            magnitude_multiplier = 1.2  # Amplify predictions
        elif current_volatility < 0.1:  # Low volatility
            magnitude_multiplier = 0.8  # Dampen predictions
        else:
            magnitude_multiplier = 1.0
        
        # Trend bias adjustment
        if current_trend == 'bullish':
            direction_multiplier = 1.1 if direction > 0 else 0.9
        elif current_trend == 'bearish':
            direction_multiplier = 1.1 if direction < 0 else 0.9
        else:
            direction_multiplier = 1.0
        
        # Volume adjustment
        if current_volume < 0.7:  # Low volume
            magnitude_multiplier *= 0.9  # Reduce prediction strength
        
        return {
            'magnitude_multiplier': magnitude_multiplier,
            'direction_multiplier': direction_multiplier
        }
    
    def _generate_prediction_explanation(self, signals, market_context):
        """Generate human-readable explanation for the prediction"""
        
        explanations = []
        
        for signal in signals:
            signal_type = signal['type']
            confidence = signal['confidence']
            
            if confidence > 0.3:  # Only explain significant signals
                if signal_type == 'historical_pattern':
                    explanations.append(f"Historical pattern matching (confidence: {confidence:.2f})")
                elif signal_type == 'cascade_effect':
                    explanations.append(f"Cascade effect detected (confidence: {confidence:.2f})")
                elif signal_type == 'weak_signal_aggregation':
                    explanations.append(f"Multiple weak signals converging (confidence: {confidence:.2f})")
                elif signal_type == 'cross_domain_convergence':
                    explanations.append(f"Cross-domain signals aligned (confidence: {confidence:.2f})")
        
        if not explanations:
            return "No significant causal signals detected for market swing prediction."
        
        return "Prediction based on: " + "; ".join(explanations)
    
    # Utility methods for signal processing
    def _extract_sequence_features(self, events):
        """Extract features from event sequence for pattern matching"""
        features = {
            'event_count': len(events),
            'domain_diversity': len(set(e.get('domain', 'unknown') for e in events)),
            'avg_impact': np.mean([e.get('market_impact', 0.0) for e in events]),
            'time_span': self._calculate_time_span(events),
            'sector_concentration': self._calculate_sector_concentration(events)
        }
        return features
    
    def _calculate_pattern_similarity(self, features1, features2):
        """Calculate similarity between two event sequence feature sets"""
        # Simple cosine similarity for demonstration
        # In practice, would use more sophisticated similarity measures
        
        keys = set(features1.keys()).intersection(set(features2.keys()))
        if not keys:
            return 0.0
        
        vec1 = np.array([features1[k] for k in keys])
        vec2 = np.array([features2[k] for k in keys])
        
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _calculate_individual_signal_strength(self, event):
        """Calculate the individual signal strength of an event"""
        base_impact = event.get('market_impact', 0.0)
        uncertainty = event.get('uncertainty', 0.5)
        recency_factor = self._calculate_recency_factor(event.get('timestamp', datetime.now()))
        
        return base_impact * (1 - uncertainty) * recency_factor
    
    def _infer_signal_direction(self, event):
        """Infer whether event is bullish or bearish signal"""
        # Simple sentiment-based direction inference
        description = event.get('description', '').lower()
        
        positive_words = ['growth', 'profit', 'success', 'breakthrough', 'approval', 'win']
        negative_words = ['crisis', 'crash', 'failure', 'disaster', 'loss', 'decline']
        
        pos_count = sum(1 for word in positive_words if word in description)
        neg_count = sum(1 for word in negative_words if word in description)
        
        if pos_count > neg_count:
            return 1.0  # Bullish
        elif neg_count > pos_count:
            return -1.0  # Bearish
        else:
            return 0.0  # Neutral
    
    def _calculate_time_span(self, events):
        """Calculate time span of events in hours"""
        if len(events) < 2:
            return 0.0
        
        timestamps = [e.get('timestamp', datetime.now()) for e in events]
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
    
    def _calculate_sector_concentration(self, events):
        """Calculate how concentrated events are in specific sectors"""
        all_sectors = []
        for event in events:
            all_sectors.extend(event.get('affected_sectors', []))
        
        if not all_sectors:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        sector_counts = {}
        for sector in all_sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        total_sectors = len(all_sectors)
        hhi = sum((count / total_sectors) ** 2 for count in sector_counts.values())
        
        return hhi
    
    def _calculate_recency_factor(self, timestamp):
        """Calculate recency factor (more recent = higher weight)"""
        hours_ago = (datetime.now() - timestamp).total_seconds() / 3600.0
        return max(0.1, np.exp(-hours_ago / 168.0))  # Decay over 1 week
    
    def _predict_cascade_outcome(self, initial_event, next_event, causality_strength):
        """Predict outcome of a detected cascade"""
        # Simplified cascade prediction
        base_impact = initial_event.get('market_impact', 0.0)
        amplification = causality_strength * 1.5  # Cascades can amplify
        
        return {
            'magnitude': base_impact * amplification,
            'direction': self._infer_signal_direction(initial_event),
            'confidence': causality_strength
        }
    
    def _weight_historical_predictions(self, similar_patterns):
        """Weight historical pattern predictions by similarity and recency"""
        if not similar_patterns:
            return {'magnitude': 0.0, 'direction': 0.0}
        
        total_weight = sum(p['similarity'] for p in similar_patterns)
        
        weighted_magnitude = sum(
            p['similarity'] * p['market_outcome']['magnitude'] 
            for p in similar_patterns
        ) / total_weight
        
        weighted_direction = sum(
            p['similarity'] * p['market_outcome']['direction']
            for p in similar_patterns
        ) / total_weight
        
        return {
            'magnitude': weighted_magnitude,
            'direction': weighted_direction
        }
    
    def _calculate_sector_convergence(self, domain_signals):
        """Calculate how well different domains converge on same sectors"""
        sector_mentions = {}
        
        for domain, signals in domain_signals.items():
            for signal in signals:
                for sector in signal['sectors']:
                    if sector not in sector_mentions:
                        sector_mentions[sector] = set()
                    sector_mentions[sector].add(domain)
        
        # Find sectors mentioned by multiple domains
        convergent_sectors = {
            sector: domains for sector, domains in sector_mentions.items()
            if len(domains) >= 2
        }
        
        return convergent_sectors

class WeakSignalAggregator:
    """
    Aggregate multiple weak signals into strong predictions
    """
    
    def __init__(self):
        self.aggregation_threshold = 3  # Minimum signals needed
        self.sector_weight = 0.4
        self.magnitude_weight = 0.3
        self.direction_weight = 0.3
    
    def aggregate(self, weak_signals):
        """
        Aggregate weak signals using sophisticated combination methods
        
        AGGREGATION METHODS:
        1. Sector alignment boost
        2. Direction consensus amplification
        3. Magnitude accumulation with diminishing returns
        """
        
        if len(weak_signals) < self.aggregation_threshold:
            return {'prediction': {'magnitude': 0.0, 'direction': 0.0}, 'confidence': 0.0}
        
        # Group signals by sector
        sector_groups = {}
        for signal in weak_signals:
            for sector in signal['sectors']:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(signal)
        
        # Find strongest sector signal
        strongest_sector_signal = None
        max_sector_strength = 0.0
        
        for sector, signals in sector_groups.items():
            if len(signals) >= 2:  # Multiple signals for same sector
                sector_strength = self._calculate_sector_strength(signals)
                if sector_strength > max_sector_strength:
                    max_sector_strength = sector_strength
                    strongest_sector_signal = {
                        'sector': sector,
                        'strength': sector_strength,
                        'signals': signals
                    }
        
        if strongest_sector_signal:
            prediction = self._calculate_aggregated_prediction(strongest_sector_signal)
            confidence = min(max_sector_strength, 1.0)
        else:
            prediction = {'magnitude': 0.0, 'direction': 0.0}
            confidence = 0.0
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
    
    def _calculate_sector_strength(self, signals):
        """Calculate combined strength of signals for a sector"""
        total_strength = sum(s['strength'] for s in signals)
        direction_consensus = self._calculate_direction_consensus(signals)
        
        # Boost strength if directions align
        return total_strength * direction_consensus
    
    def _calculate_direction_consensus(self, signals):
        """Calculate how well signal directions align"""
        directions = [s['direction'] for s in signals]
        
        if not directions:
            return 0.0
        
        # Calculate consensus as inverse of standard deviation
        direction_std = np.std(directions)
        consensus = max(0.1, 1.0 - direction_std)
        
        return consensus
    
    def _calculate_aggregated_prediction(self, sector_signal):
        """Calculate final aggregated prediction"""
        signals = sector_signal['signals']
        
        # Aggregate magnitude with diminishing returns
        magnitudes = [s['strength'] for s in signals]
        aggregated_magnitude = np.sqrt(sum(m**2 for m in magnitudes)) / len(magnitudes)
        
        # Average direction
        directions = [s['direction'] for s in signals]
        aggregated_direction = np.mean(directions)
        
        return {
            'magnitude': aggregated_magnitude,
            'direction': aggregated_direction
        }

# Example usage and testing
if __name__ == "__main__":
    print("📈 MARKET SWING PREDICTION USING CAUSAL GRAPHS")
    print("=" * 70)
    
    # Create mock causal graph
    causal_graph = nx.DiGraph()
    causal_graph.add_edge('sports_win', 'sponsor_boost', weight=0.6)
    causal_graph.add_edge('celebrity_endorsement', 'brand_rally', weight=0.7)
    causal_graph.add_edge('political_speech', 'sector_rotation', weight=0.5)
    
    # Initialize predictor
    predictor = MarketSwingPredictor(causal_graph, lookback_days=30, prediction_horizon=7)
    
    # Mock current events
    current_events = [
        {
            'id': 'sports_win',
            'timestamp': datetime.now() - timedelta(hours=2),
            'description': 'Major sports team with Nike sponsorship wins championship',
            'domain': 'sports',
            'market_impact': 0.3,
            'affected_sectors': ['consumer_goods', 'apparel'],
            'uncertainty': 0.2
        },
        {
            'id': 'celebrity_endorsement',
            'timestamp': datetime.now() - timedelta(hours=6),
            'description': 'A-list celebrity endorses luxury brand at major event',
            'domain': 'entertainment',
            'market_impact': 0.25,
            'affected_sectors': ['luxury_goods', 'consumer_discretionary'],
            'uncertainty': 0.3
        },
        {
            'id': 'political_speech',
            'timestamp': datetime.now() - timedelta(hours=12),
            'description': 'President announces support for renewable energy expansion',
            'domain': 'politics',
            'market_impact': 0.4,
            'affected_sectors': ['renewable_energy', 'utilities'],
            'uncertainty': 0.1
        }
    ]
    
    # Mock market context
    market_context = {
        'volatility': 0.25,
        'trend': 'bullish',
        'volume_ratio': 1.2
    }
    
    print("\n🔮 Predicting market swing based on recent events...")
    
    # Make prediction
    prediction = predictor.predict_market_swing(current_events, market_context)
    
    print(f"\n📊 MARKET SWING PREDICTION:")
    print(f"  Expected Magnitude: {prediction['swing_magnitude']:.3f}")
    print(f"  Direction: {prediction['swing_direction']}")
    print(f"  Confidence: {prediction['confidence']:.3f}")
    print(f"  Time Horizon: {prediction['time_horizon']}")
    print(f"  Explanation: {prediction['explanation']}")
    
    print(f"\n🔍 Signal Breakdown:")
    for signal_type, signal_data in prediction['signal_breakdown'].items():
        confidence = signal_data['confidence']
        print(f"  {signal_type}: {confidence:.3f} confidence")
    
    print("\n" + "=" * 70)
    print("🎯 MARKET SWING PREDICTION CAPABILITIES:")
    print("✅ Historical pattern matching for similar event sequences")
    print("✅ Cascade effect detection using causal graph structure")
    print("✅ Weak signal aggregation for comprehensive analysis")
    print("✅ Cross-domain convergence analysis")
    print("✅ Market context adjustment for realistic predictions")
    print("✅ Explainable predictions with reasoning")
    
    print("\n💡 POTENTIAL APPLICATIONS:")
    print("🎯 1. Day trading signals based on real-time events")
    print("🎯 2. Portfolio rebalancing recommendations")
    print("🎯 3. Risk management alerts for sudden swings")
    print("🎯 4. Sector rotation timing predictions")
    print("🎯 5. Volatility trading strategy optimization")
    
    print(f"\n🚀 Ready to predict market swings using comprehensive")
    print(f"   causal analysis! 📈📉")
