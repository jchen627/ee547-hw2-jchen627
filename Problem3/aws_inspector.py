#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import datetime as dt
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# ---------- Configs ----------
BOTO_CFG = Config(
    retries={"max_attempts": 2, "mode": "standard"},
    connect_timeout=5,
    read_timeout=10,
)
S3_OBJECT_SCAN_CAP = 10_000  # 每个桶最多扫描 1 万个对象用于“近似”统计

# ---------- Utils ----------
def iso(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, dt.datetime):
        return ts.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return str(ts)

def warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr)

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

def safe_call(fn, *args, **kwargs):
    """对 AWS API 做一次重试；捕获常见异常并返回 (ok, result|None)。"""
    try:
        return True, fn(*args, **kwargs)
    except (EndpointConnectionError, ReadTimeoutError) as e:
        # retry once
        try:
            return True, fn(*args, **kwargs)
        except Exception as e2:
            return False, e2
    except Exception as e:
        return False, e
def check_credentials():
    """验证 AWS 凭证"""
    try:
        sts = boto3.client("sts", config=BOTO_CFG)
        sts.get_caller_identity()
        return True
    except Exception as e:
        err(f"Authentication failed: {e}")
        return False

def call_with_backoff(client, func, **kwargs):
    """自动处理 Throttling 错误"""
    retries = 5
    for i in range(retries):
        try:
            return getattr(client, func)(**kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] in ["Throttling", "RequestLimitExceeded"]:
                wait = 2 ** i
                warn(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path or "")
    if parent:
        os.makedirs(parent, exist_ok=True)

def write_output_json(data, path):
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote JSON report to {os.path.abspath(path)}")
    except Exception as e:
        err(f"Failed to write JSON to '{path}': {e}")
        sys.exit(1)

def write_output_table(text, path):
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote Table report to {os.path.abspath(path)}")
    except Exception as e:
        err(f"Failed to write table to '{path}': {e}")
        sys.exit(1)

def validate_region_or_exit(region: str) -> None:
    sess = boto3.Session()
    valid = set(sess.get_available_regions("ec2"))  # 用 EC2 列表做校验
    if region not in valid:
        err(f"Invalid region '{region}'. Valid examples: us-east-1, us-west-2, ...")
        sys.exit(2)

# ---------- Collectors ----------
def get_account_info(session: boto3.Session, region: Optional[str]) -> Dict[str, Any]:
    sts = session.client("sts", config=BOTO_CFG)
    ok, res = safe_call(sts.get_caller_identity)
    if not ok:
        err(f"Authentication failed: {res}")
        sys.exit(1)
    return {
        "account_id": res["Account"],
        "user_arn": res["Arn"],
        "region": region or (session.region_name or ""),
        "scan_timestamp": iso(dt.datetime.now(dt.timezone.utc)),
    }

def list_iam_users(session: boto3.Session) -> List[Dict[str, Any]]:
    iam = session.client("iam", config=BOTO_CFG)  # IAM 是全局服务
    users: List[Dict[str, Any]] = []

    try:
        paginator = iam.get_paginator("list_users")
        for page in paginator.paginate():
            for u in page.get("Users", []):
                username = u["UserName"]
                # PasswordLastUsed 只能 get_user 拿到（也可能没有）
                last_used = ""
                ok, gu = safe_call(iam.get_user, UserName=username)
                if ok and "User" in gu and "PasswordLastUsed" in gu["User"]:
                    last_used = iso(gu["User"]["PasswordLastUsed"])

                # 附加策略（分页）
                attached = []
                try:
                    pgr = iam.get_paginator("list_attached_user_policies")
                    for p in pgr.paginate(UserName=username):
                        for pol in p.get("AttachedPolicies", []):
                            attached.append({
                                "policy_name": pol.get("PolicyName", ""),
                                "policy_arn": pol.get("PolicyArn", ""),
                            })
                except ClientError as e:
                    warn(f"Access denied listing attached policies for user {username}: {e}")

                users.append({
                    "username": username,
                    "user_id": u.get("UserId", ""),
                    "arn": u.get("Arn", ""),
                    "create_date": iso(u.get("CreateDate")),
                    "last_activity": last_used,
                    "attached_policies": attached,
                })
    except ClientError as e:
        warn(f"Access denied for IAM operations - skipping user enumeration ({e})")
    return users

def list_ec2_instances(session: boto3.Session, region: Optional[str]) -> List[Dict[str, Any]]:
    if not region:
        return []
    ec2 = session.client("ec2", region_name=region, config=BOTO_CFG)
    imgs_cache: Dict[str, str] = {}  # imageId -> name
    instances: List[Dict[str, Any]] = []
    try:
        paginator = ec2.get_paginator("describe_instances")
        for page in paginator.paginate():
            for res in page.get("Reservations", []):
                for inst in res.get("Instances", []):
                    image_id = inst.get("ImageId", "")
                    ami_name = ""
                    if image_id:
                        if image_id in imgs_cache:
                            ami_name = imgs_cache[image_id]
                        else:
                            try:
                                imgs = ec2.describe_images(ImageIds=[image_id]).get("Images", [])
                                if imgs:
                                    ami_name = imgs[0].get("Name", "")
                                imgs_cache[image_id] = ami_name
                            except ClientError:
                                # 没权限也不影响主体信息
                                pass

                    tags = {t["Key"]: t.get("Value", "") for t in inst.get("Tags", [])} if inst.get("Tags") else {}
                    sgs = [g.get("GroupId", "") for g in inst.get("SecurityGroups", [])]

                    instances.append({
                        "instance_id": inst.get("InstanceId", ""),
                        "instance_type": inst.get("InstanceType", ""),
                        "state": inst.get("State", {}).get("Name", ""),
                        "public_ip": inst.get("PublicIpAddress", "") or "-",
                        "private_ip": inst.get("PrivateIpAddress", "") or "-",
                        "availability_zone": inst.get("Placement", {}).get("AvailabilityZone", ""),
                        "launch_time": iso(inst.get("LaunchTime")),
                        "ami_id": image_id,
                        "ami_name": ami_name,
                        "security_groups": sgs,
                        "tags": tags,
                    })
    except ClientError as e:
        warn(f"Access denied for EC2 operations in {region}: {e}")
    except Exception as e:
        warn(f"Failed to list EC2 instances in {region}: {e}")
    return instances

def _format_port_range(p) -> str:
    if p.get("IpProtocol") in ("-1", "all"):
        return "all"
    from_p = p.get("FromPort")
    to_p = p.get("ToPort")
    if from_p is None or to_p is None:
        return "unknown"
    return f"{from_p}-{to_p}"

def list_security_groups(session: boto3.Session, region: Optional[str]) -> List[Dict[str, Any]]:
    if not region:
        return []
    ec2 = session.client("ec2", region_name=region, config=BOTO_CFG)
    groups: List[Dict[str, Any]] = []
    try:
        paginator = ec2.get_paginator("describe_security_groups")
        for page in paginator.paginate():
            for sg in page.get("SecurityGroups", []):
                inbound = []
                for perm in sg.get("IpPermissions", []):
                    ports = _format_port_range(perm)
                    # IPv4 sources
                    for rng in perm.get("IpRanges", []):
                        inbound.append({
                            "protocol": "all" if perm.get("IpProtocol") == "-1" else perm.get("IpProtocol"),
                            "port_range": ports,
                            "source": rng.get("CidrIp", ""),
                        })
                    # IPv6 sources
                    for rng in perm.get("Ipv6Ranges", []):
                        inbound.append({
                            "protocol": "all" if perm.get("IpProtocol") == "-1" else perm.get("IpProtocol"),
                            "port_range": ports,
                            "source": rng.get("CidrIpv6", ""),
                        })
                outbound = []
                for perm in sg.get("IpPermissionsEgress", []):
                    ports = _format_port_range(perm)
                    for rng in perm.get("IpRanges", []):
                        outbound.append({
                            "protocol": "all" if perm.get("IpProtocol") == "-1" else perm.get("IpProtocol"),
                            "port_range": ports,
                            "destination": rng.get("CidrIp", ""),
                        })
                    for rng in perm.get("Ipv6Ranges", []):
                        outbound.append({
                            "protocol": "all" if perm.get("IpProtocol") == "-1" else perm.get("IpProtocol"),
                            "port_range": ports,
                            "destination": rng.get("CidrIpv6", ""),
                        })
                groups.append({
                    "group_id": sg.get("GroupId", ""),
                    "group_name": sg.get("GroupName", ""),
                    "description": sg.get("Description", ""),
                    "vpc_id": sg.get("VpcId", ""),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound,
                })
    except ClientError as e:
        warn(f"Access denied for DescribeSecurityGroups in {region}: {e}")
    return groups

def list_s3_buckets(session: boto3.Session, region: Optional[str]) -> List[Dict[str, Any]]:
    s3 = session.client("s3", config=BOTO_CFG)
    buckets_out: List[Dict[str, Any]] = []
    try:
        ok, res = safe_call(s3.list_buckets)
        if not ok:
            raise res
        for b in res.get("Buckets", []):
            name = b["Name"]
            # 区域
            bucket_region = "us-east-1"
            try:
                lr = s3.get_bucket_location(Bucket=name)
                loc = lr.get("LocationConstraint")
                bucket_region = loc or "us-east-1"
            except ClientError as e:
                warn(f"Failed to get region for bucket '{name}': {e}")

            # 只统计与当前扫描 region 一致的桶（如果提供了 region）
            if region and bucket_region != region:
                continue

            # 近似统计对象数与大小（限制最大扫描数量，保证时延）
            obj_count = 0
            size_bytes = 0
            try:
                kwargs = {"Bucket": name, "MaxKeys": 1000}
                scanned = 0
                while True:
                    ok, resp = safe_call(s3.list_objects_v2, **kwargs)
                    if not ok:
                        raise resp
                    contents = resp.get("Contents", [])
                    for obj in contents:
                        obj_count += 1
                        size_bytes += int(obj.get("Size", 0))
                        scanned += 1
                        if scanned >= S3_OBJECT_SCAN_CAP:
                            break
                    if scanned >= S3_OBJECT_SCAN_CAP:
                        break
                    if resp.get("IsTruncated"):
                        kwargs["ContinuationToken"] = resp.get("NextContinuationToken")
                    else:
                        break
            except ClientError as e:
                warn(f"Failed to access S3 bucket '{name}': {e}")

            buckets_out.append({
                "bucket_name": name,
                "creation_date": iso(b.get("CreationDate")),
                "region": bucket_region,
                "object_count": obj_count,   # 近似
                "size_bytes": size_bytes,    # 近似
            })
    except ClientError as e:
        warn(f"Access denied for S3 list buckets: {e}")
    return buckets_out

# ---------- Output ----------
def to_table(account_info: Dict[str, Any],
             iam_users: List[Dict[str, Any]],
             ec2_instances: List[Dict[str, Any]],
             s3_buckets: List[Dict[str, Any]],
             sec_groups: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"AWS Account: {account_info.get('account_id')} ({account_info.get('region') or 'default'})")
    lines.append(f"Scan Time: {account_info.get('scan_timestamp').replace('T',' ').replace('Z',' UTC')}")
    lines.append("")

    # IAM
    lines.append(f"IAM USERS ({len(iam_users)} total)")
    lines.append(f"{'Username':20} {'Create Date':15} {'Last Activity':15} {'Policies'}")
    for u in iam_users:
        lines.append(f"{u['username'][:20]:20} {u['create_date'][:10]:15} {u['last_activity'][:10]:15} {len(u['attached_policies'])}")
    lines.append("")

    # EC2
    running = sum(1 for i in ec2_instances if i.get("state") == "running")
    stopped = sum(1 for i in ec2_instances if i.get("state") == "stopped")
    lines.append(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    lines.append(f"{'Instance ID':20} {'Type':10} {'State':10} {'Public IP':15} {'Launch Time':16}")
    for i in ec2_instances:
        launch = i['launch_time'].replace('T',' ')[:16]
        lines.append(f"{i['instance_id']:20} {i['instance_type'][:10]:10} {i['state'][:10]:10} "
                     f"{(i['public_ip'] or '-')[:15]:15} {launch}")
    lines.append("")

    # S3
    lines.append(f"S3 BUCKETS ({len(s3_buckets)} total)")
    lines.append(f"{'Bucket Name':26} {'Region':12} {'Created':12} {'Objects':8} {'Size (MB)':9}")
    for b in s3_buckets:
        mb = f"{(b['size_bytes']/1_000_000):.1f}"
        lines.append(f"{b['bucket_name'][:26]:26} {b['region'][:12]:12} {b['creation_date'][:10]:12} "
                     f"{b['object_count']:8d} {mb:>9}")
    lines.append("")

    # SG
    lines.append(f"SECURITY GROUPS ({len(sec_groups)} total)")
    lines.append(f"{'Group ID':14} {'Name':16} {'VPC ID':14} {'Inbound Rules'}")
    for g in sec_groups:
        lines.append(f"{g['group_id'][:14]:14} {g['group_name'][:16]:16} {g['vpc_id'][:14]:14} {len(g['inbound_rules'])}")
    lines.append("")
    return "\n".join(lines)

def build_json(account_info, iam_users, ec2_instances, s3_buckets, sec_groups) -> Dict[str, Any]:
    return {
        "account_info": account_info,
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": sec_groups,
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": sum(1 for i in ec2_instances if i.get("state") == "running"),
            "total_buckets": len(s3_buckets),
            "security_groups": len(sec_groups),
        }
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--format", default="json", choices=["json", "table"])
    args = ap.parse_args()

    if not check_credentials():   # <--- 新增
        sys.exit(1)

    if args.region:
        validate_region_or_exit(args.region)

    session = boto3.Session(region_name=args.region)
    account_info = get_account_info(session, args.region)

    iam_users = list_iam_users(session)
    ec2_instances = list_ec2_instances(session, args.region)
    s3_buckets = list_s3_buckets(session, args.region)
    sec_groups = list_security_groups(session, args.region)

    if args.format == "json":
        data = build_json(account_info, iam_users, ec2_instances, s3_buckets, sec_groups)
        if args.output:
            write_output_json(data, args.output)  # <--- 用新函数
        else:
            print(json.dumps(data, indent=2))
    else:
        text = to_table(account_info, iam_users, ec2_instances, s3_buckets, sec_groups)
        if args.output:
            write_output_table(text, args.output)  # <--- 用新函数
        else:
            print(text)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(f"Unhandled error: {e}")
        sys.exit(1)
