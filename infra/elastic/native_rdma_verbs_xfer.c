// Native libibverbs one-way RC QP transfer helper.
//
// TCP is used only as the control plane to exchange QP attributes. The payload
// itself is delivered by ibv_post_send/ibv_post_recv on a reliable-connected QP.

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_PORT_NUM 1
#define DEFAULT_GID_INDEX 0
#define DEFAULT_MAX_BYTES (64 * 1024 * 1024)

struct options {
    int is_server;
    const char *host;
    const char *bind_host;
    int tcp_port;
    const char *dev_name;
    int ib_port;
    int gid_index;
    const char *input_path;
    const char *output_path;
    size_t max_bytes;
};

struct qp_info {
    uint16_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid gid;
};

struct rdma_ctx {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    char *buf;
    size_t buf_size;
    struct qp_info local;
    int link_layer;
};

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static void diex(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static uint32_t random_psn(void) {
    srand((unsigned int)(time(NULL) ^ getpid()));
    return (uint32_t)(rand() & 0xffffff);
}

static void gid_to_hex(const union ibv_gid *gid, char *out) {
    for (int i = 0; i < 16; i++) {
        sprintf(out + i * 2, "%02x", gid->raw[i]);
    }
    out[32] = '\0';
}

static int hex_to_gid(const char *hex, union ibv_gid *gid) {
    if (strlen(hex) != 32) return -1;
    for (int i = 0; i < 16; i++) {
        unsigned int byte = 0;
        if (sscanf(hex + i * 2, "%02x", &byte) != 1) return -1;
        gid->raw[i] = (uint8_t)byte;
    }
    return 0;
}

static int read_line(int fd, char *buf, size_t cap) {
    size_t off = 0;
    while (off + 1 < cap) {
        char c;
        ssize_t n = recv(fd, &c, 1, 0);
        if (n <= 0) return -1;
        buf[off++] = c;
        if (c == '\n') break;
    }
    buf[off] = '\0';
    return 0;
}

static void exchange_qp_info(int fd, const struct qp_info *local, struct qp_info *remote) {
    char gid_hex[33];
    char line[256];
    gid_to_hex(&local->gid, gid_hex);
    int n = snprintf(line, sizeof(line), "%u %u %u %s\n",
                     local->lid, local->qpn, local->psn, gid_hex);
    if (send(fd, line, (size_t)n, 0) != n) die("send qp info");
    if (read_line(fd, line, sizeof(line)) != 0) diex("failed to read remote qp info");

    char remote_gid[33];
    unsigned int lid, qpn, psn;
    if (sscanf(line, "%u %u %u %32s", &lid, &qpn, &psn, remote_gid) != 4) {
        diex("invalid remote qp info");
    }
    remote->lid = (uint16_t)lid;
    remote->qpn = qpn;
    remote->psn = psn;
    if (hex_to_gid(remote_gid, &remote->gid) != 0) diex("invalid remote gid");
}

static int listen_socket(const char *bind_host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket");
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (!bind_host || strcmp(bind_host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, bind_host, &addr.sin_addr) != 1) {
        diex("bind host must be an IPv4 address");
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) die("bind");
    if (listen(fd, 1) != 0) die("listen");
    return fd;
}

static int connect_socket(const char *host, int port) {
    struct addrinfo hints, *res = NULL, *rp;
    char port_s[32];
    snprintf(port_s, sizeof(port_s), "%d", port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host, port_s, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc));
        exit(1);
    }
    int fd = -1;
    for (rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) die("connect");
    return fd;
}

static void init_rdma(struct rdma_ctx *r, const struct options *opt) {
    memset(r, 0, sizeof(*r));
    r->buf_size = opt->max_bytes;
    r->buf = (char *)calloc(1, r->buf_size);
    if (!r->buf) die("calloc");

    int num = 0;
    struct ibv_device **devs = ibv_get_device_list(&num);
    if (!devs || num == 0) diex("no RDMA devices found");
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (!opt->dev_name || strcmp(ibv_get_device_name(devs[i]), opt->dev_name) == 0) {
            dev = devs[i];
            break;
        }
    }
    if (!dev) diex("requested RDMA device not found");
    r->ctx = ibv_open_device(dev);
    if (!r->ctx) die("ibv_open_device");
    ibv_free_device_list(devs);

    struct ibv_port_attr port_attr;
    if (ibv_query_port(r->ctx, opt->ib_port, &port_attr) != 0) die("ibv_query_port");
    r->link_layer = port_attr.link_layer;

    r->pd = ibv_alloc_pd(r->ctx);
    if (!r->pd) die("ibv_alloc_pd");
    r->cq = ibv_create_cq(r->ctx, 16, NULL, NULL, 0);
    if (!r->cq) die("ibv_create_cq");
    struct ibv_qp_init_attr qpia;
    memset(&qpia, 0, sizeof(qpia));
    qpia.send_cq = r->cq;
    qpia.recv_cq = r->cq;
    qpia.qp_type = IBV_QPT_RC;
    qpia.sq_sig_all = 1;
    qpia.cap.max_send_wr = 16;
    qpia.cap.max_recv_wr = 16;
    qpia.cap.max_send_sge = 1;
    qpia.cap.max_recv_sge = 1;
    r->qp = ibv_create_qp(r->pd, &qpia);
    if (!r->qp) die("ibv_create_qp");
    r->mr = ibv_reg_mr(r->pd, r->buf, r->buf_size, IBV_ACCESS_LOCAL_WRITE);
    if (!r->mr) die("ibv_reg_mr");

    r->local.lid = port_attr.lid;
    r->local.qpn = r->qp->qp_num;
    r->local.psn = random_psn();
    if (ibv_query_gid(r->ctx, opt->ib_port, opt->gid_index, &r->local.gid) != 0) {
        die("ibv_query_gid");
    }

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = opt->ib_port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(r->qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        die("ibv_modify_qp INIT");
    }
}

static void connect_qp(struct rdma_ctx *r, const struct options *opt, const struct qp_info *remote) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote->lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = opt->ib_port;
    if (r->link_layer == IBV_LINK_LAYER_ETHERNET || remote->lid == 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote->gid;
        attr.ah_attr.grh.sgid_index = opt->gid_index;
        attr.ah_attr.grh.hop_limit = 64;
    }
    if (ibv_modify_qp(r->qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
        die("ibv_modify_qp RTR");
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = r->local.psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(r->qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
        die("ibv_modify_qp RTS");
    }
}

static void post_recv(struct rdma_ctx *r) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)r->buf;
    sge.length = (uint32_t)r->buf_size;
    sge.lkey = r->mr->lkey;
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (ibv_post_recv(r->qp, &wr, &bad) != 0) die("ibv_post_recv");
}

static size_t read_file(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) die("fopen input");
    size_t n = fread(buf, 1, cap, f);
    if (ferror(f)) die("fread");
    int extra = fgetc(f);
    fclose(f);
    if (extra != EOF) diex("input file exceeds --max-bytes");
    return n;
}

static void write_file(const char *path, const char *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) die("fopen output");
    if (fwrite(buf, 1, n, f) != n) die("fwrite");
    fclose(f);
}

static void post_send(struct rdma_ctx *r, size_t n) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)r->buf;
    sge.length = (uint32_t)n;
    sge.lkey = r->mr->lkey;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 2;
    wr.opcode = IBV_WR_SEND;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    if (ibv_post_send(r->qp, &wr, &bad) != 0) die("ibv_post_send");
}

static struct ibv_wc poll_one(struct rdma_ctx *r) {
    struct ibv_wc wc;
    int tries = 0;
    while (1) {
        int n = ibv_poll_cq(r->cq, 1, &wc);
        if (n < 0) die("ibv_poll_cq");
        if (n > 0) break;
        usleep(1000);
        if (++tries > 30000) diex("RDMA completion timeout");
    }
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "RDMA completion failed: status=%d vendor=%u\n", wc.status, wc.vendor_err);
        exit(1);
    }
    return wc;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --server --port P --out PATH [--bind IP] [--dev DEV] [--gid-index N] [--max-bytes N]\n"
            "  %s --client --host HOST --port P --in PATH [--dev DEV] [--gid-index N] [--max-bytes N]\n",
            argv0, argv0);
}

static struct options parse_args(int argc, char **argv) {
    struct options opt;
    memset(&opt, 0, sizeof(opt));
    opt.bind_host = "0.0.0.0";
    opt.ib_port = DEFAULT_PORT_NUM;
    opt.gid_index = DEFAULT_GID_INDEX;
    opt.max_bytes = DEFAULT_MAX_BYTES;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--server") == 0) opt.is_server = 1;
        else if (strcmp(argv[i], "--client") == 0) opt.is_server = 0;
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) opt.host = argv[++i];
        else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) opt.bind_host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) opt.tcp_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dev") == 0 && i + 1 < argc) opt.dev_name = argv[++i];
        else if (strcmp(argv[i], "--ib-port") == 0 && i + 1 < argc) opt.ib_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gid-index") == 0 && i + 1 < argc) opt.gid_index = atoi(argv[++i]);
        else if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) opt.input_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) opt.output_path = argv[++i];
        else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) opt.max_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else {
            usage(argv[0]);
            exit(2);
        }
    }
    if (opt.tcp_port <= 0) diex("--port is required");
    if (opt.is_server && !opt.output_path) diex("--out is required for server");
    if (!opt.is_server && (!opt.host || !opt.input_path)) diex("--host and --in are required for client");
    return opt;
}

int main(int argc, char **argv) {
    struct options opt = parse_args(argc, argv);
    struct rdma_ctx r;
    init_rdma(&r, &opt);
    if (opt.is_server) post_recv(&r);

    int control_fd;
    int listen_fd = -1;
    if (opt.is_server) {
        listen_fd = listen_socket(opt.bind_host, opt.tcp_port);
        control_fd = accept(listen_fd, NULL, NULL);
        if (control_fd < 0) die("accept");
    } else {
        control_fd = connect_socket(opt.host, opt.tcp_port);
    }

    struct qp_info remote;
    exchange_qp_info(control_fd, &r.local, &remote);
    connect_qp(&r, &opt, &remote);

    if (opt.is_server) {
        struct ibv_wc wc = poll_one(&r);
        write_file(opt.output_path, r.buf, wc.byte_len);
        char ack = 'A';
        if (send(control_fd, &ack, 1, 0) != 1) die("send ack");
    } else {
        size_t n = read_file(opt.input_path, r.buf, r.buf_size);
        post_send(&r, n);
        (void)poll_one(&r);
        char ack = 0;
        if (recv(control_fd, &ack, 1, 0) != 1 || ack != 'A') diex("missing server ack");
    }

    close(control_fd);
    if (listen_fd >= 0) close(listen_fd);
    return 0;
}
