# K-means

## Description

Implementation of k-means clustering algorithm with REST API
that receives a set of data points and returns the same set of data points with assigned
cluster label.

## Example

Request
```
curl -v -X POST -H "Content-Type: text/plain" -d "[51.1, 30.2; 64.91, 51.67; 70.45, 68.7; 61.9, 45.2]"
"https://pyalex-wix-test.herokuapp.com/clustering/labels?num_clusters=2&max_iterations=300"

> POST /clustering/labels?num_clusters=4&max_iterations=300 HTTP/1.1
> Content-Type: text/plain
> Content-Length: 51
```

Response

```
< HTTP/1.1 200 OK
< Content-Type: text/plain; charset=utf-8
<
[51.1, 30.2, 1;
64.91, 51.67, 2;
70.45, 68.7, 3;
61.9, 45.2, 4]
```