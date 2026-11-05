"""Minimal SG_IO access to an optical drive.

Sends only READ CD, SET CD SPEED and TEST UNIT READY, so nothing here can write to a disc.
The kernel accepts SET CD SPEED only on a descriptor opened for writing, hence O_RDWR.
"""
import ctypes
import fcntl
import os
import random
import struct
import time

SG_IO = 0x2285
SG_DXFER_NONE = -1
SG_DXFER_FROM_DEV = -3
BYTES_PER_SECTOR = 2352
CD_1X_KB_PER_SECOND = 176


class _Header(ctypes.Structure):
    _fields_ = [
        ("interface_id", ctypes.c_int), ("dxfer_direction", ctypes.c_int),
        ("cmd_len", ctypes.c_ubyte), ("mx_sb_len", ctypes.c_ubyte),
        ("iovec_count", ctypes.c_ushort), ("dxfer_len", ctypes.c_uint),
        ("dxferp", ctypes.c_void_p), ("cmdp", ctypes.c_void_p), ("sbp", ctypes.c_void_p),
        ("timeout", ctypes.c_uint), ("flags", ctypes.c_uint), ("pack_id", ctypes.c_int),
        ("usr_ptr", ctypes.c_void_p), ("status", ctypes.c_ubyte),
        ("masked_status", ctypes.c_ubyte), ("msg_status", ctypes.c_ubyte),
        ("sb_len_wr", ctypes.c_ubyte), ("host_status", ctypes.c_ushort),
        ("driver_status", ctypes.c_ushort), ("resid", ctypes.c_int),
        ("duration", ctypes.c_uint), ("info", ctypes.c_uint),
    ]


class Drive:
    def __init__(self, path="/dev/sr0"):
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self._sense = ctypes.create_string_buffer(64)
        self._data = ctypes.create_string_buffer(BYTES_PER_SECTOR)

    def _command(self, cdb, length=0):
        command = ctypes.create_string_buffer(bytes(cdb), len(cdb))
        header = _Header()
        header.interface_id = ord("S")
        header.dxfer_direction = SG_DXFER_FROM_DEV if length else SG_DXFER_NONE
        header.cmd_len = len(cdb)
        header.mx_sb_len = 64
        header.dxfer_len = length
        header.dxferp = ctypes.cast(self._data, ctypes.c_void_p) if length else None
        header.cmdp = ctypes.cast(command, ctypes.c_void_p)
        header.sbp = ctypes.cast(self._sense, ctypes.c_void_p)
        header.timeout = 30000
        fcntl.ioctl(self.fd, SG_IO, header)
        completed = time.perf_counter_ns()
        ok = header.status == 0 and header.host_status == 0 and header.driver_status == 0
        return ok, completed

    def read_sector(self, lba):
        """Reads one raw sector. Returns (ok, completion time in ns)."""
        cdb = [0xBE, 0] + list(struct.pack(">I", lba)) + [0, 0, 1, 0x10, 0, 0]
        return self._command(cdb, BYTES_PER_SECTOR)

    def set_cav(self, speed_x=16):
        """Requests constant rotation. The drive may pick a lower real speed; only stability matters."""
        kilobytes = speed_x * CD_1X_KB_PER_SECOND
        cdb = [0xBB, 1] + list(struct.pack(">H", kilobytes)) + [0xFF, 0xFF] + [0] * 6
        return self._command(cdb)[0]

    def require_disc(self):
        ok, _ = self._command([0, 0, 0, 0, 0, 0])
        if not ok:
            sense = self._sense.raw
            raise SystemExit(
                f"The drive is not ready: sense {sense[2] & 15:x}/{sense[12]:02x}/{sense[13]:02x}. "
                "3a means no disc, 30 means it rejected the disc, 04 means it is still loading.")


def sample_period_times(drive, lba, sector_count, count=40, hop=3000):
    """Completion times in seconds, spaced by random pauses so that only the true period folds them."""
    other = lba + hop if lba + hop < sector_count - 10 else lba - hop
    times = []
    drive.read_sector(other)
    for _ in range(count):
        time.sleep(random.random() * 0.05)
        times.append(drive.read_sector(lba)[1] / 1e9)
        drive.read_sector(other)
    return times
