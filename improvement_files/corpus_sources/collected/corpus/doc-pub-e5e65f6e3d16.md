## Looking at logs

For now, digging deeper into the cluster requires logging into the relevant machines.  Here are the locations
of the relevant log files.  On systemd-based systems, you may need to use `journalctl` instead of examining log files.

### Control Plane nodes

* `/var/log/kube-apiserver.log` - API Server, responsible for serving the API
* `/var/log/kube-scheduler.log` - Scheduler, responsible for making scheduling decisions
* `/var/log/kube-controller-manager.log` - a component that runs most Kubernetes built-in
   , with the notable exception of scheduling
  (the kube-scheduler handles scheduling).

### Worker Nodes

* `/var/log/kubelet.log` - logs from the kubelet, responsible for running containers on the node
* `/var/log/kube-proxy.log` - logs from `kube-proxy`, which is responsible for directing traffic to Service endpoints