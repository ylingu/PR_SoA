add_rules("mode.debug", "mode.release")
add_requires("eigen")

target("ml")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
    add_packages("eigen")
