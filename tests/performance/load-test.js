// Performance test script for k6
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users over 1 minute
    { duration: '3m', target: 10 },   // Stay at 10 users for 3 minutes
    { duration: '1m', target: 20 },   // Ramp up to 20 users over 1 minute
    { duration: '2m', target: 20 },   // Stay at 20 users for 2 minutes
    { duration: '1m', target: 0 },    // Ramp down to 0 users over 1 minute
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate must be below 10%
  },
};

const BASE_URL = __ENV.TEST_URL || 'http://localhost:8501';

export default function () {
  // Test homepage
  let response = http.get(`${BASE_URL}/`);
  
  check(response, {
    'Homepage loads successfully': (r) => r.status === 200,
    'Homepage response time < 2s': (r) => r.timings.duration < 2000,
    'Homepage contains title': (r) => r.body.includes('AI Trading Platform'),
  });
  
  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);
  
  sleep(1);
  
  // Test health endpoint
  response = http.get(`${BASE_URL}/_stcore/health`);
  
  check(response, {
    'Health endpoint responds': (r) => r.status === 200,
    'Health response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  errorRate.add(response.status !== 200);
  
  sleep(2);
  
  // Test dashboard page
  response = http.get(`${BASE_URL}/1_📊_Dashboard`);
  
  check(response, {
    'Dashboard loads': (r) => r.status === 200,
    'Dashboard response time < 3s': (r) => r.timings.duration < 3000,
  });
  
  errorRate.add(response.status !== 200);
  
  sleep(3);
}

// Setup function
export function setup() {
  console.log('🚀 Starting performance tests...');
  console.log(`📊 Target URL: ${BASE_URL}`);
  
  // Verify the application is running
  const response = http.get(`${BASE_URL}/_stcore/health`);
  
  if (response.status !== 200) {
    throw new Error(`Application not ready. Health check failed with status: ${response.status}`);
  }
  
  console.log('✅ Application is ready for testing');
  return { baseUrl: BASE_URL };
}

// Teardown function
export function teardown(data) {
  console.log('🏁 Performance tests completed');
  console.log(`📈 Tests ran against: ${data.baseUrl}`);
}
