<!-- overview -->
An issue that comes up rather frequently for new installations of Kubernetes is
that a Service is not working properly.  You've run your Pods through a
Deployment (or other workload controller) and created a Service, but you
get no response when you try to access it.  This document will hopefully help
you to figure out what's going wrong.

<!-- body -->

## Running commands in a Pod

For many steps here you will want to see what a Pod running in the cluster
sees.  The simplest way to do this is to run an interactive busybox Pod:

```none
kubectl run -it --rm --restart=Never busybox --image=registry.k8s.io/busybox:1.27.2 sh
```

 
If you don't see a command prompt, try pressing enter.
 

If you already have a running Pod that you prefer to use, you can run a
command in it using:

```shell
kubectl exec <POD-NAME> -c <CONTAINER-NAME> -- <COMMAND>
```