import feedparser
import re
import requests
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime

# Indian Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# Professional User-Agent header to avoid throttling
HEADERS = {"User-Agent": "AlgoTradingPlatform/1.0 (+https://github.com/trading-platform/contact)"}

class RealNewsClient:
    """Enhanced client for fetching real market news from RSS feeds with production features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Updated Indian financial news RSS feeds with reliable sources
        self.rss_feeds = {
            "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
            "Moneycontrol": "https://www.moneycontrol.com/rss/markets.xml",   # Broader markets feed
            "CNBC-TV18": "https://www.cnbctv18.com/rss/stock-market.xml",
            "BQ Prime": "https://www.bqprime.com/feed/markets",               # Updated from BloombergQuint
        }
    
    def _parse_dt(self, entry) -> Optional[datetime]:
        """
        Robustly parse datetime from RSS entry with timezone awareness
        
        Args:
            entry: RSS feed entry
            
        Returns:
            Datetime in IST timezone or None if parsing fails
        """
        # Try structured datetime fields first
        for key in ("published_parsed", "updated_parsed"):
            dt = getattr(entry, key, None)
            if dt: 
                try:
                    return datetime(*dt[:6], tzinfo=timezone.utc).astimezone(IST)
                except Exception:
                    pass
        
        # Try string datetime fields
        for key in ("published", "updated"):
            val = getattr(entry, key, None)
            if val:
                try:
                    dt = parsedate_to_datetime(val)
                    if dt.tzinfo is None: 
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(IST)
                except Exception:
                    continue
        
        return None

    def fetch_news_from_feed(self, feed_url: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch news articles from RSS feed with enhanced parsing and error handling
        
        Args:
            feed_url: RSS feed URL
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries with standardized fields
        """
        try:
            # Use professional headers to avoid throttling
            feed = feedparser.parse(feed_url, request_headers=HEADERS)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                # Parse publication date with timezone awareness
                pub_dt = self._parse_dt(entry)
                
                # Get link with fallback to id field
                link = getattr(entry, "link", None) or getattr(entry, "id", None) or ""
                
                article = {
                    "title": getattr(entry, "title", "").strip(),
                    "summary": getattr(entry, "summary", getattr(entry, "title", "")).strip(),
                    "link": link.strip(),
                    "published": pub_dt,
                    "raw_published": getattr(entry, "published", None),  # Keep original for debugging
                    "source": feed.feed.get("title", "Unknown Source"),
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"RSS fetch error: {feed_url} -> {e}")
            return []
    
    def get_latest_market_news(self, max_per_source: int = 5) -> List[Dict]:
        """
        Get latest market news from all RSS sources with deduplication and sorting
        
        Args:
            max_per_source: Maximum articles per RSS source
            
        Returns:
            Deduplicated and sorted list of articles (newest first)
        """
        all_news = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                articles = self.fetch_news_from_feed(feed_url, max_per_source)
                for article in articles:
                    article['source'] = source_name  # Override with our source name
                all_news.extend(articles)
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
                continue
        
        # Deduplicate by normalized link or title to avoid duplicates across sources
        seen = set()
        deduped = []
        for article in all_news:
            # Create a unique key from link or title (normalized)
            key = (article["link"] or article["title"]).strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(article)
        
        # Sort newest first; keep articles with unknown dates at the end
        deduped.sort(
            key=lambda x: x["published"] or datetime(1970, 1, 1, tzinfo=IST), 
            reverse=True
        )
        
        return deduped
    
    def search_news_by_symbol(self, symbol: str, news_list: List[Dict], 
                             aliases: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter news articles mentioning a specific stock symbol with enhanced matching
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            news_list: List of news articles to search
            aliases: Additional company names to search for (e.g., ['Reliance Industries', 'RIL'])
            
        Returns:
            List of relevant news articles
        """
        # Build comprehensive search terms
        terms = [symbol]
        if aliases:
            terms.extend(aliases)
        
        # Create word-boundary regex pattern to avoid false matches
        # e.g., r"\b(RELIANCE|Reliance Industries|RIL)\b"
        pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, terms)) + r")\b", 
            flags=re.IGNORECASE
        )
        
        relevant_news = []
        for article in news_list:
            # Search in both title and summary
            text = (article.get("title", "") + " " + article.get("summary", ""))
            if pattern.search(text):
                relevant_news.append(article)
        
        return relevant_news
    
    def get_recent_news(self, hours: int = 24, max_per_source: int = 10) -> List[Dict]:
        """
        Get news from the last N hours only
        
        Args:
            hours: Number of hours to look back
            max_per_source: Maximum articles per source
            
        Returns:
            List of recent news articles
        """
        cutoff = datetime.now(tz=IST) - timedelta(hours=hours)
        items = self.get_latest_market_news(max_per_source=max_per_source)
        
        # Filter to only recent articles with valid timestamps
        return [
            article for article in items 
            if article["published"] and article["published"] >= cutoff
        ]
    
    def get_news_by_symbols(self, symbols_dict: Dict[str, List[str]], 
                           max_per_source: int = 10) -> Dict[str, List[Dict]]:
        """
        Get news for multiple symbols with their aliases
        
        Args:
            symbols_dict: Dict mapping symbols to their aliases
                         e.g., {'RELIANCE': ['Reliance Industries', 'RIL'], 'TCS': ['Tata Consultancy']}
            max_per_source: Maximum articles per source
            
        Returns:
            Dictionary mapping symbols to their relevant news
        """
        all_news = self.get_latest_market_news(max_per_source)
        results = {}
        
        for symbol, aliases in symbols_dict.items():
            results[symbol] = self.search_news_by_symbol(symbol, all_news, aliases)
        
        return results
    
    def get_news_sentiment_keywords(self) -> Dict[str, List[str]]:
        """
        Get predefined sentiment keywords for basic sentiment analysis
        
        Returns:
            Dictionary with positive and negative keywords
        """
        return {
            'positive': [
                'profit', 'gain', 'rise', 'surge', 'jump', 'boost', 'strong', 'growth',
                'record', 'high', 'outperform', 'beat', 'exceed', 'rally', 'bullish',
                'upgrade', 'acquisition', 'merger', 'expansion', 'breakthrough'
            ],
            'negative': [
                'loss', 'fall', 'drop', 'decline', 'crash', 'weak', 'poor', 'slump',
                'low', 'underperform', 'miss', 'below', 'sell-off', 'bearish',
                'downgrade', 'bankruptcy', 'lawsuit', 'investigation', 'scandal'
            ]
        }
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """
        Basic sentiment analysis for a news article
        
        Args:
            article: News article dictionary
            
        Returns:
            Dictionary with sentiment analysis results
        """
        keywords = self.get_news_sentiment_keywords()
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        
        positive_count = sum(1 for word in keywords['positive'] if word in text)
        negative_count = sum(1 for word in keywords['negative'] if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "positive_score": positive_count,
            "negative_score": negative_count,
            "confidence": abs(positive_count - negative_count) / max(positive_count + negative_count, 1)
        }
