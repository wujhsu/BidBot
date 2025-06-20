#!/usr/bin/env python3
"""
测试会话修复
Test session fixes
"""

import re

def test_session_id_validation():
    """测试会话ID验证"""
    
    # 模拟前端生成的会话ID格式
    def generate_session_id():
        import time
        import random
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        return f"session_{timestamp}_{random_id}"
    
    # 后端验证函数
    def validate_session_id(session_id: str) -> bool:
        if not session_id:
            return False
        pattern = r'^session_\d+_[a-zA-Z0-9]{8,}$'
        return bool(re.match(pattern, session_id))
    
    # 测试多个会话ID
    test_cases = [
        generate_session_id(),
        generate_session_id(),
        "session_1750401161173_f20gw4rnn",  # 之前失败的ID
        "session_1234567890_abcd1234",
        "session_123_abc",  # 太短的随机部分
        "invalid_session_id",  # 无效格式
        "",  # 空字符串
    ]
    
    print("=== 会话ID验证测试 ===")
    for session_id in test_cases:
        is_valid = validate_session_id(session_id)
        status = "✅ 有效" if is_valid else "❌ 无效"
        print(f"{status}: {session_id}")
    
    print("\n=== 新生成的会话ID ===")
    for i in range(3):
        new_id = generate_session_id()
        is_valid = validate_session_id(new_id)
        status = "✅" if is_valid else "❌"
        print(f"{status} {new_id}")


def test_json_serialization():
    """测试JSON序列化"""
    from datetime import datetime
    import json
    
    print("\n=== JSON序列化测试 ===")
    
    # 测试datetime序列化
    test_data = {
        "timestamp": datetime.now(),
        "message": "测试消息"
    }
    
    try:
        # 直接序列化会失败
        json.dumps(test_data)
        print("❌ 直接序列化应该失败")
    except TypeError as e:
        print(f"✅ 预期的序列化错误: {e}")
    
    # 使用isoformat()序列化
    test_data_fixed = {
        "timestamp": datetime.now().isoformat(),
        "message": "测试消息"
    }
    
    try:
        json_str = json.dumps(test_data_fixed)
        print(f"✅ 修复后的序列化成功: {json_str}")
    except Exception as e:
        print(f"❌ 修复后的序列化失败: {e}")


if __name__ == "__main__":
    test_session_id_validation()
    test_json_serialization()
    print("\n🎉 测试完成")
