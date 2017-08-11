# DIG ETL Engine

- Manager for ETK processes and Kafka topic.
- Docker image.

## kafka input parameters of interest for Logstash
`auto_offset_resetedit`
- Value type is string
- There is no default value for this setting.

What to do when there is no initial offset in Kafka or if an offset is out of range:  
- earliest: automatically reset the offset to the earliest offset
- latest: automatically reset the offset to the latest offset
- none: throw exception to the consumer if no previous offset is found for the consumer’s group
- anything else: throw exception to the consumer.

`bootstrap_servers`
- Value type is string
- Default value is "localhost:9092"

A list of URLs to use for establishing the initial connection to the cluster. This list should be in the form of host1:port1,host2:port2 These urls are just used for the initial connection to discover the full cluster membership (which may change dynamically) so this list need not contain the full set of servers (you may want more than one, though, in case a server is down).

`consumer_threads`
- Value type is number
- Default value is 1

Ideally you should have as many threads as the number of partitions for a perfect balance — more threads than partitions means that some threads will be idle

`group_id`
- Value type is string
- Default value is "logstash"

The identifier of the group this consumer belongs to. Consumer group is a single logical subscriber that happens to be made up of multiple processors. Messages in a topic will be distributed to all Logstash instances with the same group_id

`topics`
- Value type is array
- Default value is ["logstash"]

A list of topics to subscribe to, defaults to ["logstash"].


## Docker Image

Build image

    docker build -t dig_etl_engine .
    
Run instance

    docker run -d -p 9999:9999 \
    -v $(pwd)/../mydig-projects:/projects_data \
    -v $(pwd)/config_docker_sample.py:/app/dig-etl-engine/config.py \
    dig_etl_engine