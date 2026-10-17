// CPU-only tests for the disk tier of the parked conversations (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <ctime>
#include <sys/stat.h>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
std::vector<std::string> lines;   // the tier's diagnostic lines
std::mutex lines_m;
int checks = 0;
bool g_chunked = false;   // the suite runs twice: the one-file format (v1) and the chunked one (v2)
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (%s format)\n", label, g_chunked ? "chunked" : "one-file");
        std::lock_guard<std::mutex> g(lines_m);
        for (size_t i = lines.size() > 8 ? lines.size() - 8 : 0; i < lines.size(); ++i) std::fprintf(stderr, "  log: %s\n", lines[i].c_str());
        std::exit(1);
    }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed, int32_t first = 1000) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// one stage's image: `tokens` live tokens from `first`, checkpoints at a half and a quarter of them, two K/V layers
SavedConversation image(size_t tokens, uint8_t seed, int32_t first = 1000, int64_t lo = 0, int64_t hi = 48) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.cvec = false;
    s.live = checkpoint(tokens, seed, first);
    s.checkpoints.push_back(checkpoint(tokens / 2, uint8_t(seed + 9), first));    // the deepest first
    s.checkpoints.push_back(checkpoint(tokens / 4, uint8_t(seed + 19), first));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.pooled_rows = 9;
        kv.idx_dim = 128;
        kv.k = pattern(layer == 0 ? (17u << 20) + 8192 : 4096, uint8_t(seed + layer));   // > 16 MiB: several segments (a whole number of cells)
        kv.v = pattern(4096, uint8_t(seed + 10));
        kv.k_scale = pattern(1024, uint8_t(seed + 20));
        kv.v_scale = pattern(1024, uint8_t(seed + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(seed + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// a layer split's image: the first stage's with one later stage's
SavedConversation split_image(size_t tokens, uint8_t seed, int32_t first = 1000) {
    SavedConversation s = image(tokens, seed, first, 0, 26);
    s.stage_images.push_back(image(tokens, uint8_t(seed + 50), first, 26, 48));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_images.size() != b.stage_images.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!same(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

std::vector<int32_t> prompt_of(size_t tokens, int32_t first = 1000) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < tokens; ++i) p.push_back(first + int32_t(i));
    return p;
}

size_t tier_files(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().find(ConversationDiskTier::file_prefix()) != std::string::npos) ++n;
    return n;
}

void touch(const fs::path& p) { std::ofstream(p, std::ios::binary) << "x"; }

struct FakeClock {
    ConversationDiskTier::Clock::time_point t{};
    std::function<ConversationDiskTier::Clock::time_point()> fn() { return [this] { return t; }; }
};

ConversationDiskOptions opts(const fs::path& dir, uint64_t budget, int64_t ttl_s) {
    ConversationDiskOptions o;
    o.dir = dir.string();
    o.budget_bytes = budget;
    o.ttl_s = ttl_s;
    o.chunked = g_chunked;
    o.log = [](const std::string& l) { std::lock_guard<std::mutex> g(lines_m); lines.push_back(l); };
    return o;
}
bool logged(const char* what) {
    std::lock_guard<std::mutex> g(lines_m);
    for (const auto& l : lines) if (l.find(what) != std::string::npos) return true;
    return false;
}
void chunked_suite(const fs::path& dir);

void suite(bool chunked) {
    g_chunked = chunked;
    const int before = checks;
    std::printf("conversation_disk_test: %s format\n", chunked ? "chunked (v2)" : "one-file (v1)");
    const fs::path dir = fs::temp_directory_path() / ("strata_conversation_disk_test_" + std::to_string((long long) ::getpid()) + "_" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    const uint64_t GiB = 1ull << 30;
    std::string e;

    {   // off without a folder or a budget
        ConversationDiskTier off(opts("", GiB, 1800));
        check(off.open(e) && !off.enabled() && !off.put(image(10, 1)), "no folder: off, not an error");
        ConversationDiskTier off2(opts(dir, 0, 1800));
        check(off2.open(e) && !off2.enabled(), "no budget: off");
    }

    {   // start: this tier's leftovers (and their temporary files) go, other files stay
        fs::create_directories(dir);
        touch(dir / (std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin"));
        touch(dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin.abcd.tmp"));
        touch(dir / "keep.bin");
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e) && tier.enabled(), "open");
        check(tier_files(dir) == 0 && logged("2 stale files removed"), "open removes the tier's old and temporary files");
        check(fs::exists(dir / "keep.bin"), "open keeps other files");
        fs::remove(dir / "keep.bin");
    }

    {   // background write, match, read back exactly (a layer split: two files; every checkpoint)
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "open 2");
        const SavedConversation expect = split_image(600, 1);
        check(tier.put(split_image(600, 1)), "put a split image");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 2 && !tier.pending(), "written: one file per stage");
        check(logged("wrote 600 tokens") && logged("in 2 files"), "the write is logged");
        const auto live = tier.best(prompt_of(700), {}, false);
        check(live.tokens == 600 && live.live && live.id != 0, "best: the live state as a prefix");
        const auto ck = tier.best([] { auto p = prompt_of(300); p.push_back(7); return p; }(), {}, false);
        check(ck.tokens == 300 && !ck.live, "best: the deepest checkpoint as a prefix");
        check(tier.best([] { auto p = prompt_of(150); p.push_back(7); return p; }(), {}, false).tokens == 150,
              "best: an earlier checkpoint is kept too");
        const auto whole = tier.best(prompt_of(600), {}, false);
        check(whole.tokens == 300 && !whole.live, "best: the whole prompt held is no prefix (its last token is read)");
        check(tier.best(prompt_of(700), {}, true).tokens == 0, "best: another control-vector state does not match");
        check(tier.best(prompt_of(700, 5), {}, false).tokens == 0, "best: other tokens do not match");
        SavedConversation back;
        check(tier.take(live.id, live.tokens, back, e), "take");
        check(same(back, expect), "read back exactly: both stages, the live state and every checkpoint");
        check(tier.size() == 0 && (chunked || tier.bytes() == 0) && tier_files(dir) == 0,
              "take removes the entry and its files");   // (chunked: the chunks wait out their grace)
        const auto st = tier.stats();
        check(st.queued == 1 && st.written == 1 && st.disk_hits == 1 && st.ram_hits == 0 && st.tokens_back == 600 &&
              (chunked ? st.bytes_read > (34u << 20) : st.bytes_read == st.bytes_written) &&
              st.bytes_written > (34u << 20), "counters");
        check(logged("disk hit: 600 tokens") && logged("totals:"), "the hit and the totals are logged");
    }

    {   // a queued image comes back from RAM: the writer is held at its first write
        std::promise<void> release;
        std::shared_future<void> gate = release.get_future().share();
        auto o = opts(dir, GiB, 1800);
        o.before_write = [gate] { gate.wait(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e), "open 3");
        check(tier.put(image(400, 2)), "put a (the writer takes it and waits)");
        check(tier.put(image(400, 3, 5000)), "put b (queued behind a)");
        check(tier.pending(), "pending while the writer waits");
        const auto b = tier.best(prompt_of(450, 5000), {}, false);
        check(b.tokens == 400, "b matches in the queue");
        SavedConversation back;
        check(tier.take(b.id, b.tokens, back, e) && same(back, image(400, 3, 5000)), "b back from RAM");
        check(tier.stats().ram_hits == 1 && logged("RAM hit: 400 tokens"), "a RAM hit, logged");
        release.set_value();
        tier.wait_idle();
        check(!tier.pending() && tier.size() == 1 && tier_files(dir) == 1, "a written after the release");
        check(tier.best(prompt_of(450, 5000), {}, false).tokens == 0, "b is gone from the tier");
    }

    {   // #879: clear() takes the conversations on the disk, their files too
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e) && tier.put(image(400, 40)), "clear: put a");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 1, "clear: a on the disk");
        check(tier.clear() == 1 && tier.size() == 0 && tier_files(dir) == 0 && logged("removed all 1 conversations"),
              "clear: a and its file gone");
        check(tier.best(prompt_of(450), {}, false).tokens == 0, "clear: nothing matches");
    }

    {   // the budget: the oldest goes for a new one; an image above the budget is refused
        ConversationDiskTier tier(opts(dir, 25ull << 20, 1800));   // one ~17 MiB image, not two
        check(tier.open(e), "open 4");
        check(tier.put(image(400, 4)), "first queued");
        tier.wait_idle();
        check(tier.size() == 1, "first fits");
        check(tier.put(image(500, 5, 5000)), "second queued");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().evicted == 1, "second evicts the first");
        check(tier.best(prompt_of(450), {}, false).tokens == 0 && tier.best(prompt_of(600, 5000), {}, false).tokens == 500,
              "the newer one stays");
        ConversationDiskTier small(opts(dir, 1ull << 20, 1800));
        check(small.open(e) && !small.put(image(400, 6)) && small.size() == 0, "above the budget: refused");
    }
    check(tier_files(dir) == 0, "the tier removes its files when it ends");

    {   // the time limit
        FakeClock clock;
        ConversationDiskTier tier(opts(dir, GiB, 30 * 60), clock.fn());
        check(tier.open(e), "open 5");
        check(tier.put(image(400, 7)), "put a");
        tier.wait_idle();
        clock.t += std::chrono::minutes(20);
        check(tier.put(image(400, 8, 7000)), "put b, 20 min later");
        tier.wait_idle();
        clock.t += std::chrono::minutes(9);
        check(tier.expire() == 0 && tier.size() == 2, "29 min: both kept");
        clock.t += std::chrono::minutes(2);
        check(tier.expire() == 1 && tier.size() == 1 && tier.stats().expired == 1, "31 min: the first is removed");
        check(tier.best(prompt_of(450), {}, false).tokens == 0 && tier.best(prompt_of(450, 7000), {}, false).tokens == 400,
              "the younger one stays");
        check(tier_files(dir) == 1, "the expired file is deleted");
        clock.t += std::chrono::minutes(30);
        check(tier.put(image(300, 9, 9000)), "put c, 30 min later");
        tier.wait_idle();
        check(tier.size() == 1, "a write removes the expired ones first");
    }

    {   // a read the RAM preflight refuses: nothing restored, the files are gone anyway
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e) && tier.put(split_image(400, 10)), "put for a refused read");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        const bool ok = tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; });
        check(!ok && back.kv.empty() && back.stage_images.empty() && tier.size() == 0 && tier_files(dir) == 0,
              "refused read: empty image, entry and files gone");
        check(tier.stats().read_failed == 1, "a failed read is counted");
    }

    {   // the RAM cache's evictions go to the disk tier (set_on_evict)
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "open 6");
        ConversationCache cache(GiB, 1);
        cache.set_on_evict([&](SavedConversation&& s) { tier.put(std::move(s)); });
        check(cache.put(image(400, 11)), "park a");
        check(cache.put(image(400, 12, 8000)), "park b: a leaves RAM");
        tier.wait_idle();
        check(cache.size() == 1 && cache.evictions() == 1 && tier.size() == 1, "a is on the disk");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "a matches on the disk");
        check(cache.best(prompt_of(450, 8000), {}, false).tokens == 400, "b stays in RAM");
    }

    {   // persist: the files outlive the tier, and the next tier with the same identity takes them in
        auto persistent = [&](uint64_t model) {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.identity = {model, 7};
            return o;
        };
        const std::string prefix = ConversationDiskTier::file_prefix();
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.persistent(), "persist: open");
            check(tier.put(split_image(500, 20)) && tier.put(image(400, 21, 6000)), "persist: two queued");
        }   // the end waits for the queue and keeps the files
        check(tier_files(dir) == 5 && logged("2 conversations kept on the disk"), "persist: 3 images + 2 indexes kept");
        touch(dir / ("." + prefix + "x-1-0.bin.abcd.tmp"));
        touch(dir / (prefix + "orphan-9-0.bin"));
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.size() == 2 && tier_files(dir) == 5, "persist: taken in, leftovers removed");
            check(logged("2 conversations from earlier runs"), "persist: the start is logged");
            const auto m = tier.best(prompt_of(600), {}, false);
            check(m.tokens == 500 && m.live, "persist: matches after a restart");
            SavedConversation back;
            const bool read = tier.take(m.id, m.tokens, back, e);
            if (!read) std::fprintf(stderr, "take: %s\n", e.c_str());
            check(read, "persist: read back after a restart");
            check(same(back, split_image(500, 20)), "persist: the image is exactly the one written");
            check(tier_files(dir) == 2, "persist: its images and index are gone");
        }
        {
            ConversationDiskTier other(persistent(2));   // another model, engine version or settings
            check(other.open(e) && other.size() == 0 && tier_files(dir) == 0 && logged("1 refused"),
                  "persist: another identity's files are refused and removed");
        }
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.put(image(300, 22)), "persist: put for a damaged index");
        }
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".idx") fs::resize_file(f.path(), 20);
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "persist: a damaged index is refused");
        }
        {   // strata-nvfp4: the index's hash - one byte changed, the size kept (the image byte count), is refused too
            ConversationDiskTier tier(persistent(5));
            check(tier.open(e) && tier.put(image(300, 25)), "persist: put for a flipped byte");
        }
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".idx") {
                std::fstream x(f.path(), std::ios::in | std::ios::out | std::ios::binary);
                char ch = 0;
                x.seekg(36);
                x.read(&ch, 1);
                ch = char(ch ^ 0x20);
                x.seekp(36);
                x.write(&ch, 1);
            }
        {
            ConversationDiskTier tier(persistent(5));
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "persist: a flipped index byte is refused");
        }
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.put(image(400, 23)) && tier.put(image(400, 24, 6000)),
                  "persist: put a and b, the same length");
        }
        std::vector<fs::path> bins;   // a's image first: the names differ only in the id
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".bin") bins.push_back(f.path());
        std::sort(bins.begin(), bins.end());
        check(bins.size() == 2, "persist: two images");
        fs::copy_file(bins[0], bins[1], fs::copy_options::overwrite_existing);   // b's index now names a's image
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.size() == 2, "persist: both taken in");
            const auto m = tier.best(prompt_of(450, 6000), {}, false);
            check(m.tokens == 400, "persist: b matches by its index");
            SavedConversation back;
            check(!tier.take(m.id, m.tokens, back, e) && back.kv.empty() && logged("hold another conversation"),
                  "persist: a's image under b's index is refused, never restored as b");
            check(tier.best(prompt_of(450), {}, false).tokens == 400, "persist: a is still there");
        }
    }

    // ---- strata-nvfp4 (extras/dialog-park) ----
    {   // take(keep): the entry and its files stay on the disk, as the most recently used
        ConversationDiskTier tier(opts(dir, 40ull << 20, 1800));   // two ~17 MiB images, not three
        check(tier.open(e) && tier.put(image(400, 30)) && tier.put(image(400, 31, 6000)), "keep: put a, b");
        tier.wait_idle();
        check(tier.size() == 2, "keep: a and b on the disk");
        const auto a = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(tier.take(a.id, a.tokens, back, e, {}, true) && same(back, image(400, 30)), "keep: a read back exactly");
        check(tier.size() == 2 && tier_files(dir) == 2 && tier.stats().kept == 1 && tier.has(a.id),
              "keep: a stays, with its file");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "keep: a still matches");
        check(tier.put(image(400, 39, 8000)), "keep: put c");
        tier.wait_idle();
        check(tier.has(a.id) && tier.best(prompt_of(450, 6000), {}, false).tokens == 0 &&
              tier.best(prompt_of(450, 8000), {}, false).tokens == 400,
              "keep: the budget took b (least recently used), not a (just read)");
    }
    check(tier_files(dir) == 0, "keep: the files go when the tier ends");

    {   // write_now: on the caller's thread, at once; the same keys are not written twice; a cancel leaves nothing
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "now: open");
        uint64_t id = 0, id2 = 0;
        bool cancelled = false;
        check(tier.write_now(image(400, 32), nullptr, {}, id, e, &cancelled) && id != 0 && !cancelled, "now: written");
        check(tier.size() == 1 && tier_files(dir) == 1 && tier.has(id) && !tier.pending(), "now: on the disk at once");
        check(logged("saved 400 tokens while idle"), "now: logged");
        check(tier.write_now(image(400, 32), nullptr, {}, id2, e) && id2 == id && tier.size() == 1 &&
              tier.stats().synced == 1, "now: an entry with the same keys is not written again");
        int asked = 0;
        check(!tier.write_now(image(400, 33, 7000), nullptr, [&] { return ++asked > 1; }, id2, e, &cancelled) &&
              cancelled && e == "cancelled", "now: given up at its second block");
        check(tier.size() == 1 && tier_files(dir) == 1 && tier.stats().sync_cancelled == 1,
              "now: a cancelled write leaves no file and no entry");
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(m.id == id && tier.take(m.id, m.tokens, back, e) && same(back, image(400, 32)), "now: read back exactly");
        uint64_t sid = 0;
        check(tier.write_now(split_image(300, 41), nullptr, {}, sid, e) && tier_files(dir) == 2,
              "now: a layer split's image, one file per stage");
        const auto ms = tier.best(prompt_of(350), {}, false);
        check(ms.id == sid && tier.take(ms.id, ms.tokens, back, e) && same(back, split_image(300, 41)),
              "now: the split image read back exactly");
    }

    {   // supersede: a newer copy of the same conversation removes the older one, after it is written
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "sup: open");
        uint64_t a = 0, b = 0, c = 0, d = 0;
        check(tier.write_now(image(400, 34), nullptr, {}, a, e), "sup: a turn (checkpoints 200, 100)");
        check(tier.write_now(image(400, 36, 9000), nullptr, {}, c, e), "sup: another conversation");
        check(tier.write_now(image(800, 35), nullptr, {}, b, e), "sup: the same conversation later (400, 200)");
        check(!tier.has(a) && tier.has(b) && tier.has(c) && tier.size() == 2 && tier_files(dir) == 2 &&
              tier.stats().superseded == 1, "sup: the older copy went, the other conversation stayed");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "sup: the newer copy serves the older prompt");
        tier.put(image(800, 35));   // a spilled copy with the same keys: the older one goes when it is written
        tier.wait_idle();
        check(!tier.has(b) && tier.size() == 2 && tier.stats().superseded == 2, "sup: the writer supersedes too");
        // the same conversation carried on whose own chain has rotated the old deepest checkpoint out still
        // supersedes it (its live tokens start with that checkpoint)
        {
            uint64_t x = 0, y = 0;
            check(tier.write_now(image(500, 60, 20000), nullptr, {}, x, e), "sup: X (checkpoints 250, 125)");
            SavedConversation later = image(900, 61, 20000);   // live 900 from the same tokens
            later.checkpoints.clear();
            later.checkpoints.push_back(checkpoint(800, 62, 20000));
            later.checkpoints.push_back(checkpoint(700, 63, 20000));
            check(tier.write_now(later, nullptr, {}, y, e) && !tier.has(x) && tier.has(y),
                  "sup: a rotated-out deepest checkpoint inside the newer live state: superseded");
            SavedConversation sib = image(500, 64, 20000);       // a sibling: the same first 125 tokens, then its own
            for (size_t i = 125; i < sib.live.ids.size(); ++i) sib.live.ids[i] = 90000 + int32_t(i);
            sib.checkpoints.clear();
            sib.checkpoints.push_back(checkpoint(125, 65, 20000));
            uint64_t z = 0;
            check(tier.write_now(sib, nullptr, {}, z, e) && tier.has(y) && tier.has(z),
                  "sup: a sibling that shares only the root keeps the other conversation");
        }
        // a different branch of the conversation that does not hold the deepest checkpoint stays beside it
        SavedConversation branch = image(300, 37);
        branch.checkpoints.resize(1);
        branch.checkpoints[0] = checkpoint(150, 38);
        check(tier.write_now(branch, nullptr, {}, d, e) && tier.size() == 5, "sup: another branch is kept");
    }

    {   // set_busy: the writer starts no job while the engine runs a request; wait_idle gets it written anyway
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "busy: open");
        tier.set_busy(true);
        check(tier.put(image(400, 42)), "busy: queued");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        check(tier.pending() && tier.size() == 0, "busy: not written while busy");
        tier.set_busy(false);
        for (int i = 0; i < 250 && tier.size() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(tier.size() == 1, "busy: written once idle");
        tier.set_busy(true);
        check(tier.put(image(400, 43, 7000)), "busy: queued again");
        tier.wait_idle();
        check(tier.size() == 2 && !tier.pending(), "busy: wait_idle has it written although busy");
        tier.set_busy(false);
    }

    {   // the RAM cache's disk_id: an eviction of an image already on the disk writes nothing
        ConversationDiskTier tier(opts(dir, GiB, 1800));
        check(tier.open(e), "mark: open");
        ConversationCache cache(GiB, 1);
        size_t put_calls = 0;
        cache.set_on_evict([&](SavedConversation&& s) {
            if (s.disk_id != 0 && tier.has(s.disk_id)) return;
            ++put_calls;
            tier.put(std::move(s));
        });
        check(cache.put(image(400, 44)), "mark: park a");
        uint64_t id = 0;
        check(tier.write_now(cache.at(0), nullptr, {}, id, e), "mark: a written while idle");
        cache.set_disk_id(cache.serial_at(0), id);
        check(cache.at(0).disk_id == id, "mark: a marked");
        check(cache.put(image(400, 45, 8000)), "mark: park b: a leaves RAM");
        tier.wait_idle();
        check(put_calls == 0 && tier.size() == 1 && tier.stats().queued == 0, "mark: a was not written again");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "mark: a still on the disk");
    }

    {   // shared (replicas): two engines, one folder, one pool
        fs::remove_all(dir);
        auto sharedo = [&](uint64_t model) {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.shared = true;
            o.identity = {model, 7};
            return o;
        };
        ConversationDiskTier t1(sharedo(1)), t2(sharedo(1));
        check(t1.open(e) && t2.open(e), "shared: both open");
        uint64_t a = 0, b = 0;
        check(t1.write_now(image(400, 50), nullptr, {}, a, e), "shared: engine 1 writes A");
        check(t1.rescan() == 0 && t1.size() == 1, "shared: an engine does not take in its own entry again");
        check(t2.best(prompt_of(450), {}, false).tokens == 0, "shared: engine 2 has not looked yet");
        check(t2.rescan() == 1 && t2.size() == 1 && logged("1 conversation from other engines"),
              "shared: engine 2 takes A in");
        const auto m = t2.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(m.tokens == 400 && t2.take(m.id, m.tokens, back, e, {}, true) && same(back, image(400, 50)),
              "shared: engine 2 restores A, written by engine 1, exactly");
        check(t2.write_now(image(800, 51), nullptr, {}, b, e) && t2.size() == 1 && tier_files(dir) == 2,
              "shared: engine 2's newer copy of A supersedes engine 1's (its files go)");
        check(t1.rescan() == 1 && t1.size() == 1 && t1.best(prompt_of(900), {}, false).tokens == 800 &&
              logged("1 removed by them"), "shared: engine 1 sees A's old copy gone and the new one in");
        // a foreign identity's entry and a young leftover stay at a start; an old leftover goes
        {
            ConversationDiskTier other(sharedo(9));
            check(other.open(e) && other.put(image(300, 52, 9000)), "shared: another identity writes");
        }
        const fs::path young = dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "aa-1-0.bin.1.tmp");
        const fs::path old = dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "bb-1-0.bin.2.tmp");
        touch(young);
        touch(old);
        fs::last_write_time(old, fs::file_time_type::clock::now() - std::chrono::minutes(30));
        const size_t before = tier_files(dir);
        ConversationDiskTier t3(sharedo(1));
        check(t3.open(e) && t3.size() == 1, "shared: a third engine takes in the pool");
        check(tier_files(dir) == before - 1 && fs::exists(young) && !fs::exists(old),
              "shared: another identity's files and a young leftover stay, the old leftover goes");
        fs::remove(young);
    }

    if (chunked) chunked_suite(dir);
    fs::remove_all(dir);
    std::printf("conversation_disk_test: %s format: %d checks\n", chunked ? "chunked (v2)" : "one-file (v1)", checks - before);
}

// ---- the chunked format's own tests (design 10, items 5-9 and 10's mixed folder) ----
size_t chunk_files(const fs::path& dir) {   // .chk files in every pool folder under `dir`
    size_t n = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && it->path().extension() == ".chk") ++n;
    return n;
}
std::vector<fs::path> chunk_paths(const fs::path& dir) {
    std::vector<fs::path> v;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && it->path().extension() == ".chk") v.push_back(it->path());
    std::sort(v.begin(), v.end());
    return v;
}
size_t pool_dirs(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_directory() && e.path().filename().string().rfind("strata-pool-", 0) == 0) ++n;
    return n;
}
int64_t mtime_s(const fs::path& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 ? (int64_t) st.st_mtime : -1;
}
void age(const fs::path& p, int minutes) {
    fs::last_write_time(p, fs::file_time_type::clock::now() - std::chrono::minutes(minutes));
}

// the steady clock of the tier and the wall clock of its chunk grace move together
struct Clocks {
    FakeClock steady;
    int64_t wall = (int64_t) std::time(nullptr);
    void advance(int minutes) { steady.t += std::chrono::minutes(minutes); wall += minutes * 60; }
    ConversationDiskOptions with(ConversationDiskOptions o) {
        o.wall_clock_s = [this] { return wall; };
        return o;
    }
};

// the same conversation carried on: a longer live state, the last byte of layer 0's K changed (the second slab)
SavedConversation continued(size_t tokens, uint8_t seed) {
    SavedConversation s = image(tokens, seed);
    const size_t n = s.kv[0].k.size();
    s.kv[0].k.visit(n - 1, 1, [](uint8_t* p, size_t, size_t) { *p ^= 0x5a; return true; });
    return s;
}

void chunked_suite(const fs::path& dir) {
    const uint64_t GiB = 1ull << 30;
    std::string e;
    fs::remove_all(dir);

    {   // 5. dedup: an image that differs only in its tokens writes no chunk; one changed slab writes one
        Clocks c;
        ConversationDiskTier tier(c.with(opts(dir, GiB, 1800)), c.steady.fn());
        check(tier.open(e) && pool_dirs(dir) == 1, "dedup: open makes the pool");
        uint64_t a = 0, b = 0, d = 0;
        check(tier.write_now(image(800, 70), nullptr, {}, a, e), "dedup: A written");
        const auto s0 = tier.stats();
        const size_t n0 = chunk_files(dir);
        check(s0.chunks_new > 0 && s0.new_file_bytes > (17u << 20) && n0 > 2 && n0 == tier.pool_chunks(),
              "dedup: A made its chunks");
        check(logged("chunks ") && logged("MiB deduplicated"), "dedup: the write log carries the chunk counters");
        check(tier.write_now(image(900, 70), nullptr, {}, b, e) && b != a, "dedup: the same state, longer tokens");
        const auto s1 = tier.stats();
        check(s1.chunks_new == s0.chunks_new && s1.new_file_bytes == s0.new_file_bytes && s1.chunks_hit > s0.chunks_hit &&
              s1.dedup_bytes > s0.dedup_bytes && chunk_files(dir) == n0, "dedup: an identical state writes no chunk");
        check(tier.write_now(continued(1000, 70), nullptr, {}, d, e), "dedup: one slab changed");
        const auto s2 = tier.stats();
        check(s2.chunks_new == s1.chunks_new + 1 && s2.new_file_bytes - s1.new_file_bytes < (12u << 20) &&
              chunk_files(dir) == n0 + 1, "dedup: exactly the changed slab is new");
        check(tier.summary().find("chunks: ") != std::string::npos, "dedup: the totals carry the chunk counters");
        const auto m = tier.best(prompt_of(1100), {}, false);
        SavedConversation back;
        check(m.id == d && tier.take(m.id, m.tokens, back, e) && same(back, continued(1000, 70)),
              "dedup: the continuation reads back exactly");
    }
    check(!fs::exists(dir) || pool_dirs(dir) == 0, "dedup: a non-persistent tier removes its pool at the end");

    {   // 6. take then continue: the chunks of a taken entry survive the grace and give hits; then they are swept
        Clocks c;
        ConversationDiskTier tier(c.with(opts(dir, GiB, 1800)), c.steady.fn());
        check(tier.open(e), "flaw: open");
        uint64_t a = 0, d = 0;
        check(tier.write_now(image(800, 71), nullptr, {}, a, e), "flaw: A written");
        const size_t n0 = chunk_files(dir);
        const auto m = tier.best(prompt_of(900), {}, false);
        SavedConversation back;
        check(m.id == a && tier.take(m.id, m.tokens, back, e) && same(back, image(800, 71)), "flaw: A taken");
        check(tier.size() == 0 && tier_files(dir) == 0 && chunk_files(dir) == n0 && tier.bytes() > (17u << 20),
              "flaw: the manifest is gone, the chunks are still there (ghosts)");
        c.advance(5);
        check(tier.expire() == 0 && chunk_files(dir) == n0, "flaw: 5 min: still there");
        const auto before = tier.stats();
        check(tier.write_now(continued(1000, 71), nullptr, {}, d, e), "flaw: the continuation is written");
        const auto after = tier.stats();
        check(after.chunks_new == before.chunks_new + 1 && after.chunks_hit > before.chunks_hit &&
              after.new_file_bytes - before.new_file_bytes < (12u << 20), "flaw: it hit the taken entry's chunks");
        check(chunk_files(dir) == n0 + 1, "flaw: one new chunk beside the old ones");
        c.advance(7);   // 12 min after the take: the ghost nobody took back goes, the continuation's chunks stay
        tier.expire();
        check(chunk_files(dir) == n0 && tier.pool_chunks() == n0 && logged("swept 1 unreferenced chunk"),
              "flaw: after the grace the unreferenced slab is swept, the referenced chunks stay");
        const auto m2 = tier.best(prompt_of(1100), {}, false);
        check(m2.id == d && tier.take(m2.id, m2.tokens, back, e) && same(back, continued(1000, 71)),
              "flaw: the continuation still reads back exactly");
        c.advance(11);
        tier.expire();
        check(chunk_files(dir) == 0 && tier.bytes() == 0, "flaw: nothing referenced, nothing left after the grace");
    }

    {   // 7. budget: an entry that shares everything frees about nothing and the loop goes on; TTL frees the last owner's
        Clocks c;
        ConversationDiskTier tier(c.with(opts(dir, 20ull << 20, 30 * 60)), c.steady.fn());
        check(tier.open(e), "budget: open");
        uint64_t x = 0, y = 0, z = 0;
        check(tier.write_now(image(400, 80), nullptr, {}, x, e) && tier.write_now(image(500, 80, 9000), nullptr, {}, y, e),
              "budget: two conversations with the same state");
        check(tier.size() == 2 && tier.stats().evicted == 0, "budget: dedup fits both in 20 MiB");
        check(tier.write_now(image(400, 81, 12000), nullptr, {}, z, e), "budget: a third with its own state");
        check(tier.size() == 1 && tier.has(z) && tier.stats().evicted == 2, "budget: both sharers go, the last owner frees it");
        check(tier.bytes() < (20u << 20) + (1u << 20), "budget: the pool fits again");
        c.advance(31);
        check(tier.expire() == 1 && tier.size() == 0 && chunk_files(dir) > 0, "ttl: the entry expires, its chunks wait");
        c.advance(11);
        tier.expire();
        check(chunk_files(dir) == 0, "ttl: and are swept after the grace");
    }

    {   // 8. GC: leftovers of a crash, foreign pools, hits refresh the mtime
        auto persistent = [&](uint64_t model) {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.identity = {model, 7};
            return o;
        };
        fs::remove_all(dir);
        size_t n_x = 0, n_xy = 0;
        {
            ConversationDiskTier t(persistent(1));
            check(t.open(e) && t.persistent() && pool_dirs(dir) == 1, "gc: persistent pool");
            uint64_t x = 0, y = 0;
            check(t.write_now(image(400, 82), nullptr, {}, x, e), "gc: X");
            n_x = chunk_files(dir);
            check(t.write_now(image(400, 83, 6000), nullptr, {}, y, e), "gc: Y");
            n_xy = chunk_files(dir);
            check(n_xy > n_x, "gc: Y has chunks of its own");
        }
        // a crash after Y's chunks, before its manifest: its index and manifest are gone
        std::vector<fs::path> ys;
        for (const auto& f : fs::directory_iterator(dir)) {
            const std::string name = f.path().filename().string();
            if (name.rfind(ConversationDiskTier::file_prefix(), 0) == 0 && name.find("-2") != std::string::npos) ys.push_back(f.path());
        }
        check(ys.size() == 2, "gc: Y's index and manifest found");
        for (const auto& f : ys) fs::remove(f);
        for (const auto& f : chunk_paths(dir)) age(f, 30);
        const fs::path pool = [&] {
            for (const auto& d : fs::directory_iterator(dir))
                if (d.is_directory()) return d.path();
            return fs::path();
        }();
        fs::create_directories(pool / "aa");
        touch(pool / "aa" / ".tmp-0000000000000001");
        touch(pool / "aa" / ".tmp-0000000000000002");
        age(pool / "aa" / ".tmp-0000000000000001", 30);
        {
            ConversationDiskTier t(persistent(1));
            check(t.open(e) && t.size() == 1, "gc: reopened, X taken in");
            check(chunk_files(dir) == n_x, "gc: Y's orphan chunks are swept at the start, X's stay");
            check(!fs::exists(pool / "aa" / ".tmp-0000000000000001") && fs::exists(pool / "aa" / ".tmp-0000000000000002"),
                  "gc: an old temporary chunk goes, a young one stays");
            const auto m = t.best(prompt_of(450), {}, false);
            SavedConversation back;
            check(m.tokens == 400 && t.take(m.id, m.tokens, back, e) && same(back, image(400, 82)), "gc: X reads back");
        }
        const auto known = chunk_paths(dir);
        {   // a chunk of the entry went missing: the entry is refused at the start, its files removed
            ConversationDiskTier t(persistent(1));
            check(t.open(e) && t.put(image(400, 84)), "gc: Z queued");
        }
        fs::path z;
        for (const auto& f : chunk_paths(dir))
            if (!std::binary_search(known.begin(), known.end(), f)) z = f;
        check(!z.empty(), "gc: Z has chunks of its own");
        fs::remove(z);
        {
            ConversationDiskTier t(persistent(1));
            check(t.open(e) && t.size() == 0 && tier_files(dir) == 0, "gc: a missing chunk refuses the entry");
        }
        {
            ConversationDiskTier t(persistent(2));   // another identity: the first one's pool is removed whole
            check(t.open(e) && pool_dirs(dir) == 1 && chunk_files(dir) == 0, "gc: another identity's pool is removed (only its own, empty, is left)");
        }
        fs::remove_all(dir);
        fs::create_directories(dir / "strata-pool-n0000dead0000beef" / "ab");
        touch(dir / "strata-pool-n0000dead0000beef" / "ab" / "x");
        {
            ConversationDiskTier t(opts(dir, GiB, 1800));
            check(t.open(e) && pool_dirs(dir) == 1 && !fs::exists(dir / "strata-pool-n0000dead0000beef"),
                  "gc: a non-persistent start removes the pools of earlier runs");
        }
        // a hit refreshes the mtime (at most once per grace/4)
        Clocks c;
        ConversationDiskTier t(c.with(opts(dir, GiB, 1800)), c.steady.fn());
        check(t.open(e), "gc: hit: open");
        uint64_t a = 0, b = 0;
        check(t.write_now(image(400, 85), nullptr, {}, a, e), "gc: hit: A");
        const auto cs = chunk_paths(dir);
        const int64_t t0 = mtime_s(cs.front());
        c.advance(1);
        check(t.write_now(image(500, 85), nullptr, {}, b, e) && mtime_s(cs.front()) == t0, "gc: hit: a young chunk is left alone");
        c.advance(5);
        check(t.write_now(image(600, 85), nullptr, {}, a, e) && mtime_s(cs.front()) >= c.wall - 1,
              "gc: hit: an older chunk is touched");
    }

    {   // 8b. shared: never deleted by refs=0; swept only when no manifest names it and its mtime is old
        auto sharedo = [&](Clocks& c) {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.shared = true;
            o.identity = {1, 7};
            return c.with(std::move(o));
        };
        fs::remove_all(dir);
        Clocks c;
        ConversationDiskTier t1(sharedo(c), c.steady.fn()), t2(sharedo(c), c.steady.fn());
        check(t1.open(e) && t2.open(e), "shared gc: both open");
        uint64_t x = 0, y = 0;
        check(t1.write_now(image(400, 86), nullptr, {}, x, e) && t1.write_now(image(500, 86, 9000), nullptr, {}, y, e),
              "shared gc: engine 1 writes X and Y (the same state)");
        const size_t n = chunk_files(dir);
        check(t2.rescan() == 2 && t2.size() == 2, "shared gc: engine 2 takes both in");
        auto mx = t2.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(mx.tokens == 400 && t2.take(mx.id, mx.tokens, back, e) && same(back, image(400, 86)),
              "shared gc: engine 2 takes X");
        check(chunk_files(dir) == n && t2.pool_chunks() == n, "shared gc: nothing deleted by refs=0");
        c.advance(11);
        t2.expire();
        check(chunk_files(dir) == n, "shared gc: Y's manifest still names every chunk: kept although old and unreferenced here");
        auto my = t2.best(prompt_of(550, 9000), {}, false);
        check(my.tokens == 500 && t2.take(my.id, my.tokens, back, e) && tier_files(dir) == 0, "shared gc: engine 2 takes Y too");
        c.advance(2);
        t2.expire();
        check(chunk_files(dir) == n, "shared gc: just unreferenced, mtime young: kept");
        c.advance(11);
        t2.expire();
        check(chunk_files(dir) == 0, "shared gc: unnamed and old: swept");
        check(t1.rescan() == 0 && t1.size() == 0, "shared gc: engine 1 sees both entries gone");
        t1.expire();
        check(t1.pool_chunks() == 0, "shared gc: and forgets the chunks that are gone");
    }

    {   // 9. failures: a cancel in the middle, no space, a torn chunk, two writers of one chunk
        Clocks c;
        ConversationDiskTier tier(c.with(opts(dir, GiB, 1800)), c.steady.fn());
        check(tier.open(e), "fail: open");
        uint64_t id = 0;
        int asked = 0;
        bool cancelled = false;
        check(!tier.write_now(image(400, 87), nullptr, [&] { return ++asked > 2; }, id, e, &cancelled) && cancelled &&
              e == "cancelled" && tier.size() == 0 && tier_files(dir) == 0,
              "fail: cancelled after two chunks: no manifest, no entry");
        const size_t orphans = chunk_files(dir);
        check(orphans == 2 && tier.pool_chunks() == 2, "fail: the two chunks stay as ghosts");
        c.advance(11);
        tier.expire();
        check(chunk_files(dir) == 0, "fail: the orphans are swept after the grace");
        const fs::path fdir = dir / "full";   // (a folder belongs to one non-persistent tier)
        ConversationDiskTier full([&] { auto o = c.with(opts(fdir, GiB, 1800)); o.min_free_bytes = 1ull << 60; return o; }(),
                                  c.steady.fn());
        check(full.open(e) && !full.write_now(image(400, 88), nullptr, {}, id, e) && full.size() == 0 &&
              tier_files(fdir) == 0 && chunk_files(fdir) == 0 && full.stats().write_failed == 1,
              "fail: no space for a chunk: nothing written, nothing left");
        uint64_t a = 0;
        check(tier.write_now(image(400, 89), nullptr, {}, a, e), "fail: A written");
        auto cps = chunk_paths(dir);
        fs::resize_file(cps.back(), fs::file_size(cps.back()) - 4096);   // torn: a power loss
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(m.id == a && !tier.take(m.id, m.tokens, back, e) && back.kv.empty() && tier.size() == 0 &&
              tier.stats().read_failed == 1, "fail: a torn chunk is a miss and the entry is dropped");
    }
    {
        // two engines write the same chunks at once: the no-replace rename lets one win, both succeed
        fs::remove_all(dir);
        auto sharedo = [&] {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.shared = true;
            o.identity = {1, 7};
            return o;
        };
        ConversationDiskTier t1(sharedo()), t2(sharedo());
        check(t1.open(e) && t2.open(e), "race: both open");
        uint64_t a = 0, b = 0;
        std::string e1, e2;
        bool ok1 = false, ok2 = false;
        std::thread th1([&] { ok1 = t1.write_now(image(400, 90), nullptr, {}, a, e1); });
        std::thread th2([&] { ok2 = t2.write_now(image(400, 90), nullptr, {}, b, e2); });
        th1.join();
        th2.join();
        check(ok1 && ok2, "race: both writes succeed");
        const size_t once = chunk_files(dir);
        ConversationDiskTier ref(opts(dir / "ref", GiB, 1800));
        fs::create_directories(dir / "ref");
        uint64_t r = 0;
        check(ref.open(e) && ref.write_now(image(400, 90), nullptr, {}, r, e) && chunk_files(dir / "ref") == once,
              "race: the pool holds each chunk once");
        check(t1.rescan() == 1 && t1.size() == 2, "race: engine 1 sees engine 2's entry too");
        const auto m = t1.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(m.tokens == 400 && t1.take(m.id, m.tokens, back, e, {}, true) && same(back, image(400, 90)),
              "race: the image reads back exactly");
    }

    {   // mixed folder: a v1 entry and a v2 entry side by side; v2 stays readable when a run writes v1 again
        fs::remove_all(dir);
        auto persistent = [&](bool chunked) {
            auto o = opts(dir, GiB, 0);
            o.persist = true;
            o.identity = {1, 7};
            o.chunked = chunked;
            return o;
        };
        {
            ConversationDiskTier t(persistent(false));
            check(t.open(e) && t.put(image(300, 95)), "mixed: v1 written");
        }
        check(pool_dirs(dir) == 0, "mixed: a v1 run makes no pool");
        {
            ConversationDiskTier t(persistent(true));
            check(t.open(e) && t.size() == 1, "mixed: the v1 entry is taken in by a v2 run");
            check(t.put(image(300, 96, 6000)), "mixed: v2 written beside it");
        }
        {
            ConversationDiskTier t(persistent(false));
            check(t.open(e) && t.size() == 2 && pool_dirs(dir) == 1, "mixed: both entries taken in by a v1 run");
            SavedConversation back;
            auto m1 = t.best(prompt_of(350), {}, false);
            check(m1.tokens == 300 && t.take(m1.id, m1.tokens, back, e) && same(back, image(300, 95)), "mixed: the v1 entry reads");
            auto m2 = t.best(prompt_of(350, 6000), {}, false);
            check(m2.tokens == 300 && t.take(m2.id, m2.tokens, back, e) && same(back, image(300, 96, 6000)), "mixed: the v2 entry reads");
        }
    }
}
}  // namespace

int main() {
    suite(false);
    suite(true);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
