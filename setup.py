from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai",
        "anthropic",
        "inspect_ai",
        "python-dotenv",
        "jupyter",
        "ipython",
    ],
    # extras_require={
    #     "dev": [
    #         "pytest>=7.0.0",
    #         "pytest-cov>=3.0.0",
    #         "black>=22.0.0",
    #         "flake8>=4.0.0",
    #         "mypy>=0.950",
    #     ],
    # },
    author="Lovkush Agarwal",
    author_email="lovkush@gmail.com",
    description="Package to conduct automated evaluatuions of LLMs",
    url="https://github.com/Lovkush-A/automated-evals",
    python_requires=">=3.10",
) 