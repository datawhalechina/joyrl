echo "rm -rf old dist"
rm -rf dist/
echo "rm -rf old dist"
rm -rf build/
echo "run: python setup.py sdist bdist_wheel"
python setup.py sdist bdist_wheel
echo "run: test pypi"
twine upload --repository pypitest dist/*
echo "run: upload!"
twine upload dist/*