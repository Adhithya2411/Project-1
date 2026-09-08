## Does the Service work by DNS name?

One of the most common ways that clients consume a Service is through a DNS
name.

From a Pod in the same Namespace:

```shell
nslookup hostnames
```
```none
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      hostnames
Address 1: 10.0.1.175 hostnames.default.svc.cluster.local
```

If this fails, perhaps your Pod and Service are in different
Namespaces, try a namespace-qualified name (again, from within a Pod):

```shell
nslookup hostnames.default
```
```none
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      hostnames.default
Address 1: 10.0.1.175 hostnames.default.svc.cluster.local
```

If this works, you'll need to adjust your app to use a cross-namespace name, or
run your app and Service in the same Namespace.  If this still fails, try a
fully-qualified name:

```shell
nslookup hostnames.default.svc.cluster.local
```
```none
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      hostnames.default.svc.cluster.local
Address 1: 10.0.1.175 hostnames.default.svc.cluster.local
```

Note the suffix here: "default.svc.cluster.local".  The "default" is the
Namespace you're operating in.  The "svc" denotes that this is a Service.
The "cluster.local" is your cluster domain, which COULD be different in your
own cluster.

You can also try this from a Node in the cluster:

 
10.0.0.10 is the cluster's DNS Service IP, yours might be different.
 

```shell
nslookup hostnames.default.svc.cluster.local 10.0.0.10
```
```none
Server:         10.0.0.10
Address:        10.0.0.10#53

Name:   hostnames.default.svc.cluster.local
Address: 10.0.1.175
```

If you are able to do a fully-qualified name lookup but not a relative one, you
need to check that your `/etc/resolv.conf` file in your Pod is correct.  From
within a Pod:

```shell
cat /etc/resolv.conf
```

You should see something like:

```
nameserver 10.0.0.10
search default.svc.cluster.local svc.cluster.local cluster.local example.com
options ndots:5
```

The `nameserver` line must indicate your cluster's DNS Service.  This is
passed into `kubelet` with the `--cluster-dns` flag.

The `search` line must include an appropriate suffix for you to find the
Service name.  In this case it is looking for Services in the local
Namespace ("default.svc.cluster.local"), Services in all Namespaces
("svc.cluster.local"), and lastly for names in the cluster ("cluster.local").
Depending on your own install you might have additional records after that (up
to 6 total).  The cluster suffix is passed into `kubelet` with the
`--cluster-domain` flag.  Throughout this document, the cluster suffix is
assumed to be "cluster.local".  Your own clusters might be configured
differently, in which case you should change that in all of the previous
commands.

The `options` line must set `ndots` high enough that your DNS client library
considers search paths at all.  Kubernetes sets this to 5 by default, which is
high enough to cover all of the DNS names it generates.

### Does any Service work by DNS name? {#does-any-service-exist-in-dns}

If the above still fails, DNS lookups are not working for your Service.  You
can take a step back and see what else is not working.  The Kubernetes master
Service should always work.  From within a Pod:

```shell
nslookup kubernetes.default
```
```none
Server:    10.0.0.10
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      kubernetes.default
Address 1: 10.0.0.1 kubernetes.default.svc.cluster.local
```

If this fails, please see the [kube-proxy](#is-the-kube-proxy-working) section
of this document, or even go back to the top of this document and start over,
but instead of debugging your own Service, debug the DNS Service.

## Does the Service work by IP?

Assuming you have confirmed that DNS works, the next thing to test is whether your
Service works by its IP address.  From a Pod in your cluster, access the
Service's IP (from `kubectl get` above).

```shell
for i in $(seq 1 3); do 
    wget -qO- 10.0.1.175:80
done
```

This should produce something like:

```
hostnames-632524106-bbpiw
hostnames-632524106-ly40y
hostnames-632524106-tlaok
```

If your Service is working, you should get correct responses.  If not, there
are a number of things that could be going wrong.  Read on.