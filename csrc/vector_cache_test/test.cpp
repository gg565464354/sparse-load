
std::vector<std::vector<int>> pad_and_convert_unhits_vector(
    const std::vector<std::vector<int>>& pure_unhit_list) 
{
    if (pure_unhit_list.empty()) {
        return {{0}};  // 返回 shape = [1][1]
    }

    size_t batch_size = pure_unhit_list.size();
    size_t max_unhit_len = 2;
    bool all_empty = true;

    for (const auto& unhit : pure_unhit_list) {
        if (!unhit.empty()) {
            all_empty = false;
            max_unhit_len = std::max(max_unhit_len, unhit.size());
        }
    }

    if (all_empty) {
        max_unhit_len = 1;
    }

    // 初始化结果：[max_unhit_len][batch_size]
    std::vector<std::vector<int>> result(max_unhit_len, std::vector<int>(batch_size, 0));

    // 填充
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& unhit = pure_unhit_list[b];
        for (size_t t = 0; t < unhit.size(); ++t) {
            result[t][b] = unhit[t];
        }
    }

    return result;
}
