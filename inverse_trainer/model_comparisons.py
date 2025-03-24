import numpy as np
import matplotlib.pyplot as plt

# mode = "Turkish"
mode = "English"

# Üç model ismi (dosya yollarına uyacak şekilde)
modeller = ["full_model", "pretrained_model", "pure_PPO"]

# Aynı metrik yapılarını iki kere tekrar etmeden, hem Türkçe hem İngilizce başlıkları ve verileri
# tek bir yerde saklamak için bir yapı tanımlayalım.
# Burada dictionary anahtarları (örneğin "distance_reward") hem dosya ismi hem de verileri gruplayacak.
# Her anahtarın altında "turkish_label" ve "english_label" da tutuyoruz.
metrics_config = {
    "distance_reward": {
        "turkish_label": "1) Uzaklık Ödülü",
        "english_label": "1) Distance Reward",
        "mean_values": [],
        "std_values": [],
        # Bazı metriklerde başarı oranı yok, ama yine de dictionary'de saklayıp, dolu dolu veya boş olarak tutacağız.
        "success_counts": [],
        "total_counts": []
    },
    "final_distance": {
        "turkish_label": "2) Son Mesafe",
        "english_label": "2) Final Distance",
        "mean_values": [],
        "std_values": [],
        "success_counts": [],
        "total_counts": []
    },
    "pull_success_0": {
        "turkish_label": "3) Çekme Miktarı\n(Obje >= 0 Yaklaşan Denemeler)",
        "english_label": "3) Pull Amount\n(For Trials Where Pulling >= 0)",
        "mean_values": [],
        "std_values": [],
        "success_counts": [],
        "total_counts": []
    },
    "pull_success_0.01": {
        "turkish_label": "4) Çekme Miktarı\n(Obje >= 0.01 Yaklaşan Denemeler)",
        "english_label": "4) Pull Amount\n(For Trials Where Pulling >= 0.01)",
        "mean_values": [],
        "std_values": [],
        "success_counts": [],
        "total_counts": []
    },
    "pull_success_0.05": {
        "turkish_label": "5) Çekme Miktarı\n(Obje >= 0.05 Yaklaşan Denemeler)",
        "english_label": "5) Pull Amount\n(For Trials Where Pulling >= 0.05)",
        "mean_values": [],
        "std_values": [],
        "success_counts": [],
        "total_counts": []
    }
}


for model_adi in modeller:
    dosya_yolu = f"./models/final_reward_logs/03.08_{model_adi}_500_steps_distance_and_ee_reward.npz"
    veri = np.load(dosya_yolu)

    dist_rewards = veri["dist_rewards"]
    init_dist = veri["initial_distances"]
    final_dist = veri["final_distances"]
    pull_amounts = init_dist - final_dist
    total_count = len(final_dist)

    def hesapla_cekme_orani(mask):
        count = mask.sum()
        if count > 0:
            return pull_amounts[mask].mean(), pull_amounts[mask].std(), count
        return 0.0, 0.0, count

    mean_dist_reward, std_dist_reward = dist_rewards.mean(), dist_rewards.std()
    mean_final_d, std_final_d = final_dist.mean(), final_dist.std()

    success_mask = final_dist < init_dist
    avg_success_pull, std_success_pull, success_count = hesapla_cekme_orani(success_mask)

    strict_mask_1 = pull_amounts >= 0.01
    avg_0_01_success, std_0_01_success, strict_count_1 = hesapla_cekme_orani(strict_mask_1)

    strict_mask_2 = pull_amounts >= 0.05
    avg_0_05_success, std_0_05_success, strict_count_2 = hesapla_cekme_orani(strict_mask_2)

    print(f"Model: {model_adi}")
    print(f"  - (Obje >= 0 Yaklaşan) Başarı: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"  - (Obje >= 0.01 Yaklaşan) Başarı: {strict_count_1}/{total_count} ({strict_count_1/total_count*100:.1f}%)")
    print(f"  - (Obje >= 0.05 Yaklaşan) Başarı: {strict_count_2}/{total_count} ({strict_count_2/total_count*100:.1f}%)")
    print("----------------------------------------------------")

    # Her modelin hesapladığı bu beş metrik: distance_reward, final_distance,
    # pull_success_0, pull_success_0.01, pull_success_0.05.
    # Sırasıyla yakaladığımız değerlere ekleyelim.

    # 1) distance_reward
    metrics_config["distance_reward"]["mean_values"].append(mean_dist_reward)
    metrics_config["distance_reward"]["std_values"].append(std_dist_reward)
    # distance_reward'a özel başarı sayıları yok ama dictionary'de alan var, doldurmayacağız.

    # 2) final_distance
    metrics_config["final_distance"]["mean_values"].append(mean_final_d)
    metrics_config["final_distance"]["std_values"].append(std_final_d)

    # 3) pull_success_0
    metrics_config["pull_success_0"]["mean_values"].append(avg_success_pull)
    metrics_config["pull_success_0"]["std_values"].append(std_success_pull)
    metrics_config["pull_success_0"]["success_counts"].append(success_count)
    metrics_config["pull_success_0"]["total_counts"].append(total_count)

    # 4) pull_success_0.01
    metrics_config["pull_success_0.01"]["mean_values"].append(avg_0_01_success)
    metrics_config["pull_success_0.01"]["std_values"].append(std_0_01_success)
    metrics_config["pull_success_0.01"]["success_counts"].append(strict_count_1)
    metrics_config["pull_success_0.01"]["total_counts"].append(total_count)

    # 5) pull_success_0.05
    metrics_config["pull_success_0.05"]["mean_values"].append(avg_0_05_success)
    metrics_config["pull_success_0.05"]["std_values"].append(std_0_05_success)
    metrics_config["pull_success_0.05"]["success_counts"].append(strict_count_2)
    metrics_config["pull_success_0.05"]["total_counts"].append(total_count)

# Şimdi grafik oluşturma aşaması
x_pozisyonlari = np.array([0.2, 0.5, 0.8])

# metrics_config üzerinde sırayla ilerlerken, plot isimlerini sabitlemek için bir sıralı liste oluşturalım.
plot_order = [
    ("distance_reward", "distance_reward"),
    ("final_distance", "final_distance"),
    ("pull_success_0", "pull_success_0"),
    ("pull_success_0.01", "pull_success_0.01"),
    ("pull_success_0.05", "pull_success_0.05")
]

for chunk_key, filename in plot_order:
    chunk_dict = metrics_config[chunk_key]
    if mode == "Turkish":
        chunk_adi = chunk_dict["turkish_label"]
    else:
        chunk_adi = chunk_dict["english_label"]

    y_means = np.array(chunk_dict["mean_values"])
    y_stds = np.array(chunk_dict["std_values"])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Soldaki eksen: asıl metrik (yüzdeye çevirelim)
    y_means *= 100
    y_stds *= 100

    min_y = min(y_means - y_stds)
    max_y = max(y_means + y_stds) * 1.1
    ax.set_ylim(0 if min_y < 0 else min_y, max_y)

    ax.errorbar(
        x_pozisyonlari,
        y_means,
        yerr=[np.minimum(y_stds, y_means), y_stds],
        fmt='o',
        capsize=14,
        elinewidth=1,
        markeredgewidth=1
    )
    ax.set_xticks(x_pozisyonlari)
    if mode == "Turkish":
        ax.set_xticklabels(["Tam Model", "Ön Eğitimli Model", "PPO"], rotation=15, ha='right', fontsize=12)
    else:
        ax.set_xticklabels(["Full Model", "Pretrained Model", "PPO"], rotation=15, ha='right', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_title(chunk_adi, fontsize=12)

    # Sağdaki eksende başarı oranları bar grafiği çizecek miyiz?
    # Başarı ile ilgili sayılar varsa success_counts anahtarından anlarız.
    if any(chunk_dict["success_counts"]):
        success_counts = np.array(chunk_dict["success_counts"])
        total_counts = np.array(chunk_dict["total_counts"])
        success_rates = np.where(total_counts > 0, success_counts / total_counts, 0) * 100

        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Başarı Oranı (%)" if mode == "Turkish" else "Success Rate (%)")

        bar_positions = x_pozisyonlari + 0.07
        bar_width = 0.1

        ax2.bar(
            bar_positions,
            success_rates,
            alpha=0.3,
            width=bar_width
        )

    # Kaydetme
    if mode == "Turkish":
        plt.savefig(f"comparisons_{filename}.pdf", bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f"comparisons_{filename}_english.pdf", bbox_inches='tight', transparent=True)

    plt.close(fig)