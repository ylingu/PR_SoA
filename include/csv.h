#ifndef CSV_H
#define CSV_H

#include <map>
#include <string>
#include <vector>

/**
 * @class CSV
 * @brief A class for handling CSV files.
 *
 * This class provides functions for reading and writing CSV files, as well as
 * manipulating the data within them.
 */
class CSV {
private:
    std::string filepath_;                        ///< The path to the CSV file.
    std::vector<std::vector<std::string>> data_;  ///< The data in the CSV file.
    std::map<std::string, int>
        column_names_;  ///< A map from column names to their indices.
    bool has_header_;   ///< Whether the CSV file has a header row.

private:
    auto UpdateColumnNames() -> void;
    auto ReadCSV() -> void;
    auto WriteCSV() -> void;

public:
    /**
     * @brief Constructs a new CSV object.
     *
     * @param filepath specifies the path of an existing CSV-file
     * to populate the Document data with.
     * @param has_header Whether the CSV file has a header row.
     */
    CSV(const std::string &filepath, bool has_header = true)
        : filepath_(filepath),
          has_header_(has_header),
          data_(),
          column_names_() {
        if (!filepath.empty()) {
            ReadCSV();
        }
    }

    /**
     * @brief Read Document data from file.
     *
     * @param filepath specifies the path of an existing CSV-file
     * to populate the Document data with.
     * @param has_header Whether the CSV file has a header row.
     */
    auto Load(const std::string &filepath = std::string(),
              bool has_header = true) -> void;

    /**
     * @brief Write Document data to file.
     * @param filepath optionally specifies the path where the CSV-file will be
     * created (if not specified, the original path provided when creating or
     * loading the Document data will be used).
     */
    auto Save(const std::string &filepath = std::string()) -> void;

    /**
     * @brief   Clears loaded Document data.
     */
    auto Clear() -> void;

    /**
     * @brief Gets the index of a column.
     *
     * @param column_name The name of the column.
     * @return The index of the column.
     */
    auto GetColumnIndex(const std::string &column_name) const -> int;

    /**
     * @brief Gets a column by name.
     *
     * @param column_name The name of the column.
     * @return The column data.
     */
    auto GetColumn(const std::string &column_name) const
        -> std::vector<std::string>;

    /**
     * @brief Gets a column by index.
     *
     * @param column_index The index of the column.
     * @return The column data.
     */
    auto GetColumn(const int column_index) const -> std::vector<std::string>;

    /**
     * @brief Sets a column by name.
     *
     * @param column_name The name of the column.
     * @param column_data The new data for the column.
     */
    auto SetColumn(const std::string &column_name,
                   const std::vector<std::string> &column_data) -> void;

    /**
     * @brief Sets a column by index.
     *
     * @param column_index The index of the column.
     * @param column_data The new data for the column.
     */
    auto SetColumn(const int column_index,
                   const std::vector<std::string> &column_data) -> void;

    /**
     * @brief Removes a column by name.
     *
     * @param column_name The name of the column to remove.
     */
    auto RemoveColumn(const std::string &column_name) -> void;

    /**
     * @brief Removes a column by index.
     *
     * @param column_index The index of the column to remove.
     */
    auto RemoveColumn(const int column_index) -> void;

    /**
     * @brief Inserts a column at a specific index.
     *
     * @param column_index The index at which to insert the new column.
     * @param column_data The data for the new column.
     */
    auto InsertColumn(const int column_index,
                      const std::vector<std::string> &column_data) -> void;

    /**
     * @brief Gets the number of columns in the CSV file.
     *
     * @return The number of columns in the CSV file.
     */
    inline auto GetColumnCount() const -> int { return data_[0].size(); };

    /**
     * @brief Gets a row by index.
     *
     * @param row_index The index of the row.
     * @return The row data.
     */
    auto GetRow(const int row_index) const -> std::vector<std::string>;

    /**
     * @brief Sets a row by index.
     *
     * @param row_index The index of the row.
     * @param row_data The new data for the row.
     */
    auto SetRow(const int row_index, const std::vector<std::string> &row_data)
        -> void;

    /**
     * @brief Removes a row by index.
     *
     * @param row_index The index of the row to remove.
     */
    auto RemoveRow(const int row_index) -> void;

    /**
     * @brief Inserts a row at a specific index.
     *
     * @param row_index The index at which to insert the new row.
     * @param row_data The data for the new row.
     */
    auto InsertRow(const int row_index,
                   const std::vector<std::string> &row_data) -> void;

    /**
     * @brief Gets the number of rows in the CSV file.
     *
     * @return The number of rows in the CSV file.
     */
    inline auto GetRowCount() const -> int { return data_.size(); }

    /**
     * @brief Gets a cell by row and column index.
     *
     * @param row_index The index of the row.
     * @param column_index The index of the column.
     * @return The cell data.
     */
    auto GetCell(const int row_index, const int column_index) const
        -> std::string;

    /**
     * @brief Gets a cell by row index and column name.
     *
     * @param row_index The index of the row.
     * @param column_name The name of the column.
     * @return The cell data.
     */
    auto GetCell(const int row_index, const std::string &column_name) const
        -> std::string;

    /**
     * @brief Sets a cell by row and column index.
     *
     * @param row_index The index of the row.
     * @param column_index The index of the column.
     * @param cell_data The new data for the cell.
     */
    auto SetCell(const int row_index, const int column_index,
                 const std::string &cell_data) -> void;

    /**
     * @brief Sets a cell by row index and column name.
     *
     * @param row_index The index of the row.
     * @param column_name The name of the column.
     * @param cell_data The new data for the cell.
     */
    auto SetCell(const int row_index, const std::string &column_name,
                 const std::string &cell_data) -> void;

    /**
     * @brief Gets the name of a column by index.
     *
     * @param column_index The index of the column.
     * @return The name of the column.
     */
    auto GetColumnName(const int column_index) const -> std::string;

    /**
     * @brief Gets the names of all columns.
     *
     * @return A vector of column names.
     */
    auto GetColumnNames() const -> std::vector<std::string>;
};

#endif  // CSV_H