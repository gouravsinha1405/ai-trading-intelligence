#!/bin/bash

echo "🚀 Setting up AI Trading Platform with Next.js + FastAPI"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup Backend
echo "📦 Setting up FastAPI backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Backend dependencies installed"

# Setup Frontend
echo "📦 Setting up Next.js frontend..."
cd ../frontend

# Install Node.js dependencies
npm install

echo "✅ Frontend dependencies installed"

# Create startup scripts
cd ..

# Backend startup script
cat > start-backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting FastAPI backend..."
cd backend
source venv/bin/activate
python main.py
EOF

chmod +x start-backend.sh

# Frontend startup script
cat > start-frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Next.js frontend..."
cd frontend
npm run dev
EOF

chmod +x start-frontend.sh

# Full startup script
cat > start-platform.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Trading Platform..."
echo "Backend will start on http://localhost:8000"
echo "Frontend will start on http://localhost:3000"
echo "================================"

# Start backend in background
./start-backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start frontend
./start-frontend.sh &
FRONTEND_PID=$!

echo "✅ Platform started successfully!"
echo "📊 Dashboard: http://localhost:3000"
echo "🔧 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the platform, press Ctrl+C"

# Wait for user interrupt
trap "echo 'Stopping platform...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
EOF

chmod +x start-platform.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Start the platform: ./start-platform.sh"
echo "2. Open http://localhost:3000 in your browser"
echo "3. Backend API docs: http://localhost:8000/docs"
echo ""
echo "🔧 Development:"
echo "- Backend only: ./start-backend.sh"
echo "- Frontend only: ./start-frontend.sh"
echo "- Full platform: ./start-platform.sh"
