#!/usr/bin/env python3
"""
Quick start script for Polish Bankruptcy MLOps project
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required tools are installed"""
    required_tools = ["python", "docker", "docker-compose"]

    for tool in required_tools:
        try:
            result = subprocess.run(
                [tool, "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"✅ {tool} is installed: {result.stdout.strip().split()[0:2]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"❌ {tool} is not installed or not in PATH")
            return False

    return True


def setup_environment():
    """Setup the development environment"""
    logger.info("Setting up development environment...")

    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        logger.warning("⚠️  Not in a virtual environment. Consider creating one:")
        logger.warning("   python -m venv venv")
        logger.warning(
            "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
        )
    else:
        logger.info("✅ Virtual environment detected")

    # Install dependencies
    try:
        logger.info("Installing Python dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        logger.info("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

    return True


def setup_data():
    """Setup initial data"""
    logger.info("Checking data setup...")

    data_dir = Path("data/processed")
    bankruptcy_file = data_dir / "bankruptcy_combined.csv"

    if bankruptcy_file.exists():
        logger.info("✅ Processed data found")
        return True
    else:
        logger.info("📊 Processed data not found. You may need to run data preparation:")
        logger.info("   python scripts/prepare_data.py")
        return True


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        if env_example.exists():
            logger.info("📝 Creating .env file from template...")
            import shutil

            shutil.copy(env_example, env_file)
            logger.info("✅ .env file created")
            logger.warning(
                "⚠️  Please edit .env file with your actual values before running services"
            )
        else:
            logger.warning("⚠️  .env.example not found. Creating basic .env file...")
            with open(env_file, "w") as f:
                f.write(
                    """# Basic environment configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_change_me
POSTGRES_DB=bankruptcy_db
MLFLOW_TRACKING_URI=http://localhost:5001
API_SECRET_KEY=change_this_secret_key_in_production
"""
                )
            logger.info("✅ Basic .env file created")
    else:
        logger.info("✅ .env file already exists")


def run_tests():
    """Run basic tests"""
    logger.info("Running tests...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("✅ All tests passed")
            return True
        else:
            logger.warning("⚠️  Some tests failed:")
            logger.warning(result.stdout)
            logger.warning(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Tests timed out")
        return False
    except FileNotFoundError:
        logger.warning("⚠️  pytest not found. Skipping tests.")
        return False


def start_services():
    """Start Docker services"""
    logger.info("Starting Docker services...")

    try:
        # Check if Docker is running
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        logger.info("✅ Docker is running")
    except subprocess.CalledProcessError:
        logger.error("❌ Docker is not running. Please start Docker and try again.")
        return False

    try:
        logger.info("Building and starting services...")
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "infrastructure/docker/docker-compose.yml",
                "up",
                "--build",
                "-d",
            ],
            check=True,
        )
        logger.info("✅ Services started successfully")

        logger.info("📡 Services available at:")
        logger.info("   - API: http://localhost:8000")
        logger.info("   - MLflow: http://localhost:5000")
        logger.info("   - API Docs: http://localhost:8000/docs")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start services: {e}")
        return False


def main():
    """Main function"""
    logger.info("🚀 Starting Polish Bankruptcy MLOps Quick Setup...")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    logger.info(f"📂 Working directory: {project_root.absolute()}")

    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed. Please install missing tools.")
        return 1

    # Setup environment
    if not setup_environment():
        logger.error("❌ Environment setup failed.")
        return 1

    # Create .env file
    create_env_file()

    # Setup data
    setup_data()

    # Run tests (optional)
    logger.info("🧪 Testing the setup...")
    run_tests()  # Don't fail if tests fail

    # Ask user if they want to start services
    try:
        response = (
            input("\n🐳 Do you want to start Docker services now? (y/n): ")
            .lower()
            .strip()
        )
        if response in ["y", "yes"]:
            if start_services():
                logger.info("\n🎉 Setup completed successfully!")
                logger.info("🔗 Next steps:")
                logger.info("   1. Edit .env file with your configuration")
                logger.info(
                    "   2. Train a model: python scripts/train_pipeline.py --model-type xgboost"
                )
                logger.info("   3. Visit http://localhost:8000/docs to explore the API")
                logger.info("   4. Check MLflow at http://localhost:5000")
            else:
                logger.error("❌ Service startup failed.")
                return 1
        else:
            logger.info("⏭️  Skipping service startup.")
            logger.info(
                "   To start services later: docker-compose -f infrastructure/docker/docker-compose.yml up -d"
            )
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup interrupted by user.")
        return 0

    logger.info("\n✅ Quick setup completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
