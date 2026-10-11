import type { Chunk, Collection, OutlineItem } from "./types.ts";

// Sample slice of the Kubernetes docs used by the demo collection.
const KB = "https://kubernetes.io/docs/concepts/";
const U = {
  life: KB + "workloads/pods/pod-lifecycle/",
  dep: KB + "workloads/controllers/deployment/",
  svc: KB + "services-networking/service/",
  sec: KB + "configuration/secret/",
  cm: KB + "configuration/configmap/",
  res: KB + "configuration/manage-resources-containers/",
  pv: KB + "storage/persistent-volumes/",
};

const K8S_RAW: [string, string[], string][] = [
  [U.life, ["Pods", "Pod lifecycle", "Pod phase"], "A Pod's status field is a PodStatus object, which has a phase field. The phase of a Pod is a high-level summary of where the Pod is in its lifecycle. The phase values are Pending, Running, Succeeded, Failed and Unknown."],
  [U.life, ["Pods", "Container probes", "Liveness probe"], "The kubelet uses liveness probes to know when to restart a container. For example, a liveness probe can catch a deadlock, where an application is running but unable to make progress. Restarting a container in such a state can help to make the application more available despite bugs."],
  [U.life, ["Pods", "Container probes", "Readiness probe"], "The kubelet uses readiness probes to know when a container is ready to start accepting traffic. When a Pod is not ready, it is removed from Service load balancers. A readiness probe that fails does not restart the container."],
  [U.life, ["Pods", "Container probes", "Startup probe"], "A startup probe verifies whether the application within a container is started. It can be used on slow starting containers so the kubelet does not kill them before they are up and running. If a startup probe is configured, liveness and readiness probes do not start until it succeeds."],
  [U.life, ["Pods", "Pod termination"], "When a Pod is deleted, the kubelet sends SIGTERM to each container and waits for terminationGracePeriodSeconds, 30 seconds by default, before sending SIGKILL."],
  [U.dep, ["Deployments", "Updating a Deployment", "Rolling update"], "A Deployment's rollout is triggered if and only if the Deployment's Pod template is changed. Other updates, such as scaling the Deployment, do not trigger a rollout. The rolling update strategy is controlled by maxUnavailable and maxSurge, which both default to 25%."],
  [U.dep, ["Deployments", "Rolling back a Deployment"], "Sometimes you may want to roll back a Deployment, for example when it is crash looping or not stable. Run `kubectl rollout undo deployment/nginx-deployment` to return to the previous revision. Use `--to-revision=2` to roll back to a specific revision."],
  [U.dep, ["Deployments", "Failed Deployment"], "A Deployment can get stuck trying to deploy its newest ReplicaSet without ever completing. Kubernetes marks it with a ProgressDeadlineExceeded condition once `.spec.progressDeadlineSeconds`, 600 seconds by default, elapses. Check progress with `kubectl rollout status deployment/nginx-deployment`."],
  [U.dep, ["Deployments", "Clean up policy", "Revision history"], "`.spec.revisionHistoryLimit` sets how many old ReplicaSets are retained to allow rollback. By default 10 old ReplicaSets are kept. Setting it to 0 cleans up all history, and you cannot undo a rollout."],
  [U.dep, ["Deployments", "Scaling a Deployment"], "You can scale a Deployment with `kubectl scale deployment/nginx-deployment --replicas=10`. If horizontal Pod autoscaling is enabled, use `kubectl autoscale` to set minimum and maximum replicas based on CPU utilization."],
  [U.svc, ["Services", "Service types", "ClusterIP"], "ClusterIP exposes the Service on a cluster-internal IP. Choosing this value makes the Service only reachable from within the cluster. This is the default Service type."],
  [U.svc, ["Services", "Service types", "NodePort"], "NodePort exposes the Service on each Node's IP at a static port. Kubernetes allocates a port from the range 30000-32767 by default."],
  [U.sec, ["Configuration", "Secrets"], "A Secret is an object that contains a small amount of sensitive data such as a password, a token, or a key. Secrets are stored base64-encoded and are unencrypted by default in etcd, so enable encryption at rest to protect them."],
  [U.cm, ["Configuration", "ConfigMaps"], "A ConfigMap is an API object used to store non-confidential data in key-value pairs. Pods can consume ConfigMaps as environment variables, command-line arguments, or as configuration files in a volume."],
  [U.res, ["Resource management", "Requests and limits"], "If a container exceeds its memory limit, it becomes a candidate for termination and is reported as OOMKilled. A container that exceeds its CPU limit is throttled, not killed."],
  [U.pv, ["Storage", "PersistentVolumes"], "A PersistentVolume is a piece of storage in the cluster provisioned by an administrator or dynamically using a StorageClass. A PersistentVolumeClaim is a request for storage by a user."],
];

export const idOf = (i: number) => "c" + String(i + 1).padStart(2, "0");

export function outlineFrom(chunks: Chunk[]): OutlineItem[] {
  const seen = new Set<string>();
  const out: OutlineItem[] = [];
  chunks.forEach((c) =>
    c.path.forEach((_, d) => {
      const k = c.path.slice(0, d + 1).join("/");
      if (!seen.has(k)) {
        seen.add(k);
        out.push({ d, label: c.path[d], id: c.id });
      }
    }),
  );
  return out;
}

const K8S_CHUNKS: Chunk[] = K8S_RAW.map((r, i) => ({ id: idOf(i), page: r[0], url: r[0], path: r[1], text: r[2], chars: r[2].length }));

export const K8S_COL: Collection = { id: "k8s", name: "Kubernetes docs", chunks: K8S_CHUNKS, outline: outlineFrom(K8S_CHUNKS), total: 17730, docs: 1652 };
