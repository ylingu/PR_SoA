#include "csv.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

auto CSV::UpdateColumnNames() -> void {
    column_names_.clear();
    if (has_header_) {
        for (int i = 0; i < data_[0].size(); i++) {
            column_names_[data_[0][i]] = i;
        }
    }
}

auto CSV::ReadCSV() -> void {
    std::ifstream stream(filepath_);
    if (!stream.is_open()) {
        throw std::runtime_error("File not found");
    }
    std::string line;
    while (std::getline(stream, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data_.push_back(row);
    }
    stream.close();
    UpdateColumnNames();
}

auto CSV::WriteCSV() -> void {
    std::ofstream stream(filepath_);
    if (!stream.is_open()) {
        throw std::runtime_error("File not found");
    }
    for (const auto &row : data_) {
        for (auto it = row.begin(); it != row.end(); it++) {
            stream << *it;
            if (it != row.end() - 1) stream << ",";
        }
        stream << std::endl;
    }
    stream.close();
}

auto CSV::Load(const std::string &filepath, bool has_header) -> void {
    filepath_ = filepath;
    has_header_ = has_header;
    data_.clear();
    ReadCSV();
}

auto CSV::Save(const std::string &filepath) -> void {
    if (!filepath.empty()) {
        filepath_ = filepath;
    }
    WriteCSV();
}

auto CSV::Clear() -> void {
    data_.clear();
    column_names_.clear();
}

auto CSV::GetColumnIndex(const std::string &column_name) const -> int {
    if (has_header_) {
        if (column_names_.find(column_name) != column_names_.end()) {
            return column_names_.at(column_name);
        }
    }
    return -1;
}

auto CSV::GetColumn(const std::string &column_name) const
    -> std::vector<std::string> {
    auto index = GetColumnIndex(column_name);
    return GetColumn(index);
}

auto CSV::GetColumn(const int column_index) const -> std::vector<std::string> {
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    std::vector<std::string> column;
    for (auto it = data_.begin(); it != data_.end(); it++) {
        column.push_back((*it)[column_index]);
    }
    return column;
}

auto CSV::SetColumn(const std::string &column_name,
                    const std::vector<std::string> &column_data) -> void {
    auto index = GetColumnIndex(column_name);
    SetColumn(index, column_data);
}

auto CSV::SetColumn(const int column_index,
                    const std::vector<std::string> &column_data) -> void {
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    if (column_data.size() != data_.size()) {
        throw std::invalid_argument("Data size not match");
    }
    for (int i = 0; i < column_data.size(); i++) {
        data_[i][column_index] = column_data[i];
    }
    if (has_header_) {
        UpdateColumnNames();
    }
}

auto CSV::RemoveColumn(const std::string &column_name) -> void {
    auto index = GetColumnIndex(column_name);
    RemoveColumn(index);
}

auto CSV::RemoveColumn(const int column_index) -> void {
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    for (auto it = data_.begin(); it != data_.end(); it++) {
        it->erase(it->begin() + column_index);
    }
    if (has_header_) {
        UpdateColumnNames();
    }
}

auto CSV::InsertColumn(const int column_index,
                       const std::vector<std::string> &column_data) -> void {
    if (column_index < 0 || column_index > data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    if (column_data.size() != data_.size()) {
        throw std::invalid_argument("Data size not match");
    }
    for (int i = 0; i < column_data.size(); i++) {
        data_[i].insert(data_[i].begin() + column_index, column_data[i]);
    }
    if (has_header_) {
        UpdateColumnNames();
    }
}

auto CSV::GetRow(const int row_index) const -> std::vector<std::string> {
    if (row_index < 0 || row_index >= data_.size()) {
        throw std::out_of_range("Row not found");
    }
    return data_[row_index];
}

auto CSV::SetRow(const int row_index, const std::vector<std::string> &row_data)
    -> void {
    if (row_index < 0 || row_index >= data_.size()) {
        throw std::out_of_range("Row not found");
    }
    if (row_data.size() != data_[0].size()) {
        throw std::invalid_argument("Data size not match");
    }
    data_[row_index] = row_data;
}

auto CSV::RemoveRow(const int row_index) -> void {
    if (row_index < 0 || row_index >= data_.size()) {
        throw std::out_of_range("Row not found");
    }
    if (has_header_ && row_index == 0) {
        throw std::invalid_argument("Cannot remove header");
    }
    data_.erase(data_.begin() + row_index);
}

auto CSV::InsertRow(const int row_index,
                    const std::vector<std::string> &row_data) -> void {
    if (row_index < 0 || row_index > data_.size()) {
        throw std::out_of_range("Row not found");
    }
    if (row_data.size() != data_[0].size()) {
        throw std::invalid_argument("Data size not match");
    }
    if (has_header_ && row_index == 0) {
        throw std::invalid_argument("Cannot insert header");
    }
    data_.insert(data_.begin() + row_index, row_data);
}

auto CSV::GetCell(const int row_index, const int column_index) const
    -> std::string {
    if (row_index < 0 || row_index >= data_.size()) {
        throw std::out_of_range("Row not found");
    }
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    return data_[row_index][column_index];
}

auto CSV::GetCell(const int row_index, const std::string &column_name) const
    -> std::string {
    auto index = GetColumnIndex(column_name);
    return GetCell(row_index, index);
}

auto CSV::SetCell(const int row_index, const int column_index,
                  const std::string &cell_data) -> void {
    if (row_index < 0 || row_index >= data_.size()) {
        throw std::out_of_range("Row not found");
    }
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    data_[row_index][column_index] = cell_data;
    if (has_header_ && row_index == 0) {
        UpdateColumnNames();
    }
}

auto CSV::SetCell(const int row_index, const std::string &column_name,
                  const std::string &cell_data) -> void {
    auto index = GetColumnIndex(column_name);
    SetCell(row_index, index, cell_data);
}

auto CSV::GetColumnName(const int column_index) const -> std::string {
    if (column_index < 0 || column_index >= data_[0].size()) {
        throw std::out_of_range("Column not found");
    }
    if (has_header_) {
        return data_[0][column_index];
    }
    return std::to_string(column_index);
}

auto CSV::GetColumnNames() const -> std::vector<std::string> {
    if (has_header_) {
        return data_[0];
    }
    throw std::runtime_error("No header");
}