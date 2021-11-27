aws s3 cp s3://drivendata-competition-fb-isc-data-asia/all/query_images/ /facebook/data/images/query --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data-asia/all/reference_images/ /facebook/data/images/reference_1M_root/reference_1M --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data-asia/all/training_images/ /facebook/data/images/train1M/train --recursive --no-sign-request
<<COMMENT
#phase2 query download
aws s3 cp s3://drivendata-competition-fb-isc-data-asia/all/phase2_query_images/ /facebook/data/images/query --recursive --no-sign-request
COMMENT

