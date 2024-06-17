add_rules("mode.debug", "mode.release")
set_languages("cxx23")
add_requires("opencv", "eigen", "zlib")

target("library")
    set_kind("shared")
    add_files("src/*.cc")
    add_includedirs("include", {public = true})
    add_packages("opencv", "eigen")

target("main")
    add_files("*.cc")
    add_packages("opencv", "eigen")
    add_deps("library")

for i = 1, 5 do
    target("exp" .. i)
        add_files("examples/exp" .. i .. ".cc")
        add_packages("opencv", "eigen", "zlib")
        add_deps("library")
end