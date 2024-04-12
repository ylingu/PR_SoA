#ifndef CSV_H
#define CSV_H

#include <map>
#include <string>
#include <vector>

class CSV {
private:
    std::string filepath_;
    std::vector<std::vector<std::string>> data_;
    std::map<std::string, int> column_names_;
    bool has_header_;

private:
    auto UpdateColumnNames() -> void;
    auto ReadCSV() -> void;
    auto WriteCSV() -> void;

public:
    CSV(const std::string &filepath, bool has_header = true)
        : filepath_(filepath),
          has_header_(has_header),
          data_(),
          column_names_() {
        if (!filepath.empty()) {
            ReadCSV();
        }
    }
    auto Load(const std::string &filepath = std::string(),
              bool has_header = true) -> void;
    auto Save(const std::string &filepath = std::string()) -> void;
    auto Clear() -> void;
    auto GetColumnIndex(const std::string &column_name) const -> int;
    auto GetColumn(const std::string &column_name) const
        -> std::vector<std::string>;
    auto GetColumn(const int column_index) const -> std::vector<std::string>;
    auto SetColumn(const std::string &column_name,
                   const std::vector<std::string> &column_data) -> void;
    auto SetColumn(const int column_index,
                   const std::vector<std::string> &column_data) -> void;
    auto RemoveColumn(const std::string &column_name) -> void;
    auto RemoveColumn(const int column_index) -> void;
    auto InsertColumn(const int column_index,
                      const std::vector<std::string> &column_data) -> void;
    auto GetColumnCount() const -> int;
    auto GetRow(const int row_index) const -> std::vector<std::string>;
    auto SetRow(const int row_index, const std::vector<std::string> &row_data)
        -> void;
    auto RemoveRow(const int row_index) -> void;
    auto InsertRow(const int row_index,
                   const std::vector<std::string> &row_data) -> void;
    auto GetRowCount() const -> int;
    auto GetCell(const int row_index, const int column_index) const
        -> std::string;
    auto GetCell(const int row_index, const std::string &column_name) const
        -> std::string;
    auto SetCell(const int row_index, const int column_index,
                 const std::string &cell_data) -> void;
    auto SetCell(const int row_index, const std::string &column_name,
                 const std::string &cell_data) -> void;
    auto GetColumnName(const int column_index) const -> std::string;
    auto GetColumnNames() const -> std::vector<std::string>;
};

#endif