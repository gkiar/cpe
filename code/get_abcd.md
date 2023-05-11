
### Data download instructions

1. Get a Scipy container running

   ```
   docker run -ti scipy/scipy-dev
   ```

2. Install necessary packages inside container
   ```
   pip install nda-tools
   pip install secretstorage --upgrade keyrings.alt
   ```

3. Run data download tool
   ```
   downloadcmd -dp ####### -wt 5 -s3 s3://cmi-abcd-..../path
   ```
