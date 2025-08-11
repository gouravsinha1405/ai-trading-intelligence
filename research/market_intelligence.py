"""
Phase 1: Market Intelligence Research Framework

This module provides tools for systematic research and analysis of:
1. Academic literature in financial ML
2. Competitive analysis of existing platforms
3. Alternative data source evaluation
4. Open source model assessment
5. Gap identification for alpha generation opportunities
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sys
import os
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY
from openai import OpenAI

@dataclass
class ResearchPaper:
    """Structure for academic paper information"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: Optional[str]
    published_date: str
    categories: List[str]
    url: str
    relevance_score: float = 0.0
    key_insights: List[str] = None

@dataclass
class CompetitorAnalysis:
    """Structure for competitor platform analysis"""
    name: str
    url: str
    business_model: str
    key_features: List[str]
    pricing: Dict[str, str]
    target_market: str
    strengths: List[str]
    weaknesses: List[str]
    differentiation_opportunity: str

@dataclass
class DataSource:
    """Structure for alternative data source information"""
    name: str
    data_type: str
    api_endpoint: Optional[str]
    cost: str
    update_frequency: str
    coverage: str
    potential_alpha: str
    implementation_difficulty: str

class MarketIntelligenceEngine:
    """
    Comprehensive market intelligence gathering system
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.research_data = {
            'papers': [],
            'competitors': [],
            'data_sources': [],
            'insights': []
        }
    
    def search_arxiv_papers(self, query: str, max_results: int = 50) -> List[ResearchPaper]:
        """
        Search arXiv for recent financial ML papers
        """
        print(f"🔍 Searching arXiv for: {query}")
        
        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        
        # Search query - focusing on recent papers (last 2 years)
        search_query = f"({query}) AND submittedDate:[{(datetime.now() - timedelta(days=730)).strftime('%Y%m%d')}* TO *]"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                try:
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name = author.find('{http://www.w3.org/2005/Atom}name').text
                        authors.append(name)
                    
                    # Extract arXiv ID and URL
                    arxiv_url = entry.find('{http://www.w3.org/2005/Atom}id').text
                    arxiv_id = arxiv_url.split('/')[-1]
                    
                    # Extract published date
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text
                    
                    # Extract categories
                    categories = []
                    for category in entry.findall('{http://arxiv.org/schemas/atom}category'):
                        categories.append(category.get('term'))
                    
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        arxiv_id=arxiv_id,
                        published_date=published,
                        categories=categories,
                        url=arxiv_url
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"⚠️  Error parsing paper: {e}")
                    continue
            
            print(f"✅ Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            print(f"❌ Error searching arXiv: {e}")
            return []
    
    def analyze_paper_relevance(self, paper: ResearchPaper) -> ResearchPaper:
        """
        Use AI to analyze paper relevance and extract key insights
        """
        prompt = f"""
        Analyze this financial ML research paper for practical trading applications:
        
        Title: {paper.title}
        Abstract: {paper.abstract[:1000]}...
        
        Rate this paper's relevance for systematic trading (0-100) and provide:
        1. Relevance score (0-100)
        2. Key practical insights (3-5 bullet points)
        3. Implementation difficulty (1-10)
        4. Potential alpha generation capability
        5. Data requirements
        
        Focus on PRACTICAL applications, not theoretical contributions.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            # Extract relevance score (basic parsing)
            lines = analysis.split('\n')
            for line in lines:
                if 'relevance' in line.lower() and any(char.isdigit() for char in line):
                    try:
                        score = float(''.join(filter(str.isdigit, line)))
                        paper.relevance_score = min(score, 100)
                        break
                    except:
                        paper.relevance_score = 50  # Default
            
            # Extract insights
            paper.key_insights = [line.strip() for line in lines if line.strip().startswith('-') or line.strip().startswith('•')]
            
            return paper
            
        except Exception as e:
            print(f"⚠️  Error analyzing paper relevance: {e}")
            paper.relevance_score = 0
            return paper
    
    def research_competitors(self) -> List[CompetitorAnalysis]:
        """
        Research existing quantitative trading platforms
        """
        print("🏢 Researching competitor platforms...")
        
        # Known competitors to analyze
        competitors_to_research = [
            "QuantConnect", "Quantopian", "Zipline", "WorldQuant", 
            "Numerai", "Kaggle Competitions", "Alpha Architect",
            "QuantRocket", "Blueshift", "CloudQuant"
        ]
        
        competitors = []
        
        for competitor_name in competitors_to_research:
            try:
                # Use AI to research each competitor
                research_prompt = f"""
                Research the quantitative trading platform: {competitor_name}
                
                Provide a comprehensive analysis including:
                1. Business model and revenue streams
                2. Key features and capabilities
                3. Pricing structure
                4. Target market (retail vs institutional)
                5. Main strengths
                6. Notable weaknesses or limitations
                7. Market positioning
                8. Opportunities for differentiation
                
                Focus on factual, current information about their actual offerings.
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": research_prompt}],
                    max_tokens=800
                )
                
                analysis = response.choices[0].message.content
                
                # Basic parsing to structure the response
                competitor = CompetitorAnalysis(
                    name=competitor_name,
                    url=f"https://{competitor_name.lower().replace(' ', '')}.com",
                    business_model="SaaS/Platform",  # Default
                    key_features=[],
                    pricing={},
                    target_market="Unknown",
                    strengths=[],
                    weaknesses=[],
                    differentiation_opportunity=analysis[-200:]  # Last part often contains opportunities
                )
                
                competitors.append(competitor)
                print(f"✅ Analyzed {competitor_name}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Error researching {competitor_name}: {e}")
        
        return competitors
    
    def identify_alternative_data_sources(self) -> List[DataSource]:
        """
        Identify and evaluate alternative data sources
        """
        print("📡 Identifying alternative data sources...")
        
        # Categories of alternative data to research
        data_categories = [
            "Social media sentiment", "Satellite imagery", "Weather data",
            "Economic indicators", "News sentiment", "Corporate filings",
            "Supply chain data", "Patent filings", "Job postings",
            "Web scraping data", "Mobile app usage", "Credit card transactions"
        ]
        
        data_sources = []
        
        for category in data_categories:
            try:
                research_prompt = f"""
                Research free and low-cost data sources for: {category}
                
                For each data source, provide:
                1. Name and description
                2. API endpoint or access method
                3. Cost structure (focus on free/cheap options)
                4. Update frequency
                5. Data coverage and quality
                6. Potential for alpha generation in trading
                7. Implementation difficulty (1-10)
                8. Specific use cases for quantitative trading
                
                Prioritize sources that are:
                - Free or very low cost
                - Have APIs or structured access
                - Update frequently
                - Have proven track record in finance
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": research_prompt}],
                    max_tokens=600
                )
                
                sources_info = response.choices[0].message.content
                
                # Create data source entry
                source = DataSource(
                    name=f"{category} Sources",
                    data_type=category,
                    api_endpoint=None,
                    cost="Mixed",
                    update_frequency="Varies",
                    coverage="Global",
                    potential_alpha=sources_info[:200],
                    implementation_difficulty="Medium"
                )
                
                data_sources.append(source)
                print(f"✅ Researched {category}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️  Error researching {category}: {e}")
        
        return data_sources
    
    def evaluate_open_source_models(self) -> Dict[str, str]:
        """
        Evaluate open source ML models for financial applications
        """
        print("🔧 Evaluating open source models...")
        
        model_categories = [
            "Time series forecasting", "Sentiment analysis", "Anomaly detection",
            "Portfolio optimization", "Risk management", "Feature engineering",
            "Backtesting frameworks", "Real-time data processing"
        ]
        
        evaluations = {}
        
        for category in model_categories:
            try:
                eval_prompt = f"""
                Evaluate the best open source models/libraries for: {category} in quantitative trading
                
                For each tool/library, analyze:
                1. Name and GitHub repository
                2. Maturity and maintenance status
                3. Performance and accuracy
                4. Ease of integration
                5. Documentation quality
                6. Community support
                7. Real-world usage in finance
                8. Pros and cons
                
                Rank the top 3-5 options and explain why they're superior to commercial alternatives.
                Focus on tools that are actively maintained and have proven results.
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=700
                )
                
                evaluation = response.choices[0].message.content
                evaluations[category] = evaluation
                
                print(f"✅ Evaluated {category}")
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Error evaluating {category}: {e}")
        
        return evaluations
    
    def generate_opportunity_analysis(self) -> str:
        """
        Generate comprehensive opportunity analysis
        """
        print("💡 Generating opportunity analysis...")
        
        analysis_prompt = """
        Based on current quantitative trading landscape in 2025, identify the top 5 unexploited opportunities for a new AI-powered trading platform.
        
        Consider:
        1. Gaps in existing platforms
        2. Emerging technologies not yet widely adopted
        3. Alternative data sources being underutilized
        4. Market inefficiencies that can be exploited
        5. Regulatory changes creating new opportunities
        6. Retail vs institutional market needs
        
        For each opportunity:
        - Describe the opportunity
        - Estimate market size
        - Technical requirements
        - Competitive advantage potential
        - Implementation timeline
        - Revenue potential
        
        Focus on opportunities that can realistically be pursued by a small, agile team with strong ML capabilities.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️  Error generating opportunity analysis: {e}")
            return "Analysis generation failed"
    
    def run_comprehensive_research(self) -> Dict:
        """
        Execute the complete Phase 1 research pipeline
        """
        print("🚀 Starting comprehensive market intelligence research...")
        
        # 1. Academic paper research
        print("\n" + "="*50)
        print("📚 ACADEMIC RESEARCH")
        print("="*50)
        
        financial_ml_queries = [
            "machine learning finance trading",
            "deep learning portfolio optimization", 
            "time series forecasting financial markets",
            "sentiment analysis stock prediction",
            "reinforcement learning algorithmic trading"
        ]
        
        all_papers = []
        for query in financial_ml_queries:
            papers = self.search_arxiv_papers(query, max_results=20)
            for paper in papers[:5]:  # Analyze top 5 from each category
                analyzed_paper = self.analyze_paper_relevance(paper)
                all_papers.append(analyzed_paper)
            time.sleep(2)  # Be respectful to arXiv
        
        # 2. Competitor analysis
        print("\n" + "="*50)
        print("🏢 COMPETITOR ANALYSIS")
        print("="*50)
        competitors = self.research_competitors()
        
        # 3. Alternative data research
        print("\n" + "="*50)
        print("📡 ALTERNATIVE DATA RESEARCH")
        print("="*50)
        data_sources = self.identify_alternative_data_sources()
        
        # 4. Open source model evaluation
        print("\n" + "="*50)
        print("🔧 OPEN SOURCE MODEL EVALUATION")
        print("="*50)
        model_evaluations = self.evaluate_open_source_models()
        
        # 5. Opportunity analysis
        print("\n" + "="*50)
        print("💡 OPPORTUNITY ANALYSIS")
        print("="*50)
        opportunities = self.generate_opportunity_analysis()
        
        # Compile results
        research_results = {
            'papers': all_papers,
            'competitors': competitors,
            'data_sources': data_sources,
            'model_evaluations': model_evaluations,
            'opportunities': opportunities,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n🎉 Research complete! Generating summary report...")
        return research_results
    
    def save_research_results(self, results: Dict, filename: str = None):
        """
        Save research results to file
        """
        if filename is None:
            filename = f"research/market_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'papers':
                serializable_results[key] = [
                    {
                        'title': p.title,
                        'authors': p.authors,
                        'abstract': p.abstract[:500] + "..." if len(p.abstract) > 500 else p.abstract,
                        'arxiv_id': p.arxiv_id,
                        'published_date': p.published_date,
                        'categories': p.categories,
                        'url': p.url,
                        'relevance_score': p.relevance_score,
                        'key_insights': p.key_insights or []
                    } for p in value
                ]
            elif key == 'competitors':
                serializable_results[key] = [
                    {
                        'name': c.name,
                        'url': c.url,
                        'business_model': c.business_model,
                        'key_features': c.key_features,
                        'pricing': c.pricing,
                        'target_market': c.target_market,
                        'strengths': c.strengths,
                        'weaknesses': c.weaknesses,
                        'differentiation_opportunity': c.differentiation_opportunity
                    } for c in value
                ]
            elif key == 'data_sources':
                serializable_results[key] = [
                    {
                        'name': d.name,
                        'data_type': d.data_type,
                        'api_endpoint': d.api_endpoint,
                        'cost': d.cost,
                        'update_frequency': d.update_frequency,
                        'coverage': d.coverage,
                        'potential_alpha': d.potential_alpha,
                        'implementation_difficulty': d.implementation_difficulty
                    } for d in value
                ]
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📄 Research results saved to: {filename}")
        return filename

if __name__ == "__main__":
    # Initialize research engine
    research = MarketIntelligenceEngine()
    
    # Run comprehensive research
    results = research.run_comprehensive_research()
    
    # Save results
    research.save_research_results(results)
    
    print("\n🎯 Phase 1 Market Intelligence Complete!")
    print("Next: Analyze results and plan Phase 2 implementation.")
